#!/usr/bin/env python3

import subprocess
import wandb
import argparse
import json
import time
import traceback

parser = argparse.ArgumentParser(description='Run benchmark')
parser.add_argument('--branch', type=str, help='branch of ChatRWKV', default='main')
parser.add_argument('--model', type=str, help='Model path', required=True)
parser.add_argument('--verbose', action='store_true', help='Print command output')
parser.add_argument('-n', type=int, help='Number of runs', required=True)
args = parser.parse_args()

models = [args.model]

strategies = ['fp16', 'fp32', 'fp16i8']

columns = ['Device'] + strategies

local_device = '2080'

vast_id = {}

vast_dev_names = {'1080': 'GTX_1080', '2080': 'RTX_2080', '3080': 'RTX_3080', '4090': 'RTX_4090'}


class NoInstanceError(RuntimeError):
    pass


def prepare_vastai_env(device: str):
    vast_device_name = vast_dev_names[device]
    output = check_output(["vastai", "search", "offers", f"gpu_name={vast_device_name} cuda_vers>=11.8", "--raw"], args.verbose)
    output = json.loads(output)
    if len(output) == 0:
        raise NoInstanceError(f"No Vast.ai offers found for {device}")
    best = output[0]["id"]
    print(f"Found best offer {best}")
    output = check_output(f"vastai create instance {best} --image daquexian/cuda-pytorch:cu118-dev-2.0.1 --disk 32 --raw".split(), args.verbose)
    output = json.loads(output)
    instance_id = output["new_contract"]
    print(f"Created instance {instance_id}, checking status..")
    flag = False
    while not flag:
        time.sleep(10)
        print("Checking status..")
        # too verbose
        output = check_output(f"vastai show instances --raw".split(), False)
        output = json.loads(output)
        for instance in output:
            if instance["id"] == instance_id:
                print(f"Instance {instance_id} is {instance['actual_status']}")
                if instance["actual_status"] == "running":
                    vast_id[device] = (f'root@{instance["ssh_host"]}', instance["ssh_port"], instance_id)
                    flag = True
                    # sleep for a while to make sure the instance is ready
                    time.sleep(5)
                    break

    ssh_prefix = f'ssh -o StrictHostKeyChecking=no -p {vast_id[device][1]} {vast_id[device][0]}'.split()
    check_output(ssh_prefix + 'git clone https://github.com/BlinkDL/ChatRWKV'.split(), args.verbose)
    if args.branch != 'main':
        if '/' in args.branch:
            user, branch = args.branch.split('/')
            check_output(ssh_prefix + [f'cd ChatRWKV && git remote add daquexian https://github.com/{user}/ChatRWKV && git fetch {user}'], args.verbose)
        check_output(ssh_prefix + [f'cd ChatRWKV && git checkout {args.branch}'], args.verbose)
    check_output(ssh_prefix + 'pip install numpy'.split(), args.verbose)
    check_output(ssh_prefix + 'apt install ninja-build'.split(), args.verbose)

    scp('v2/benchmark-me.py', f'ChatRWKV/v2/benchmark-me.py', vast_id[device][0], vast_id[device][1])
    return ssh_prefix


wandb.init()

table = wandb.Table(columns=columns)


def scp(src, dst, dst_ip, dst_port):
    print(f"scp from {src} to {dst} of {dst_ip}:{dst_port}")
    subprocess.run(['scp', '-o', 'StrictHostKeyChecking=no', '-P', str(dst_port), src, f'{dst_ip}:{dst}'], stderr=subprocess.STDOUT)


def check_output(command, print_output):
    print(f'Running {" ".join(command)}')
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = ""
    for line in proc.stdout:
        if print_output:
            print(line.decode('utf-8').strip())
        stdout += line.decode('utf-8')
    assert proc.wait() == 0, f"Command {' '.join(command)} failed with stdout {stdout}"
    return stdout.strip()


for device in ['4090', '3080', '1080', 'cpu', '2080']:
    if device in ['cpu', local_device]:
        ssh_prefix = []
        ssh_dir = ''
    else:
        try:
            ssh_prefix = prepare_vastai_env(device)
        except NoInstanceError:
            print(f"No instance found for {device}, skipping")
            continue
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()
        ssh_dir = 'ChatRWKV/'
    device_type = 'cpu' if device == 'cpu' else 'cuda'
    for model in models:
        if device in vast_id:
            scp(model, f'ChatRWKV/{model}', vast_id[device][0], vast_id[device][1])
        data = [device]
        for strategy in strategies:
            try:
                latency = 99999999999
                for _ in range(args.n):
                    command = [*ssh_prefix, 'python3', f'{ssh_dir}v2/benchmark-me.py', '--model', f'{ssh_dir}{model}', '--strategy', f'{device_type}@{strategy}', '--custom-cuda-op', '--jit', '--only-slow']
                    print(f'Running: {" ".join(command)}')
                    output = check_output(command, print_output=args.verbose)
                    latency = min(latency, float(output.splitlines()[-2].split(' ')[2][:-2]))
                    mem = float(output.splitlines()[-1].split(' ')[-2])
                data.append(f'{latency * 1000:.0f}ms/{mem:.0f}MB') # type: ignore[reportUnboundVariable]
            except:
                data.append('N/A')
                print(f'Failed to run {model} on {device} with {strategy}')
        table.add_data(*data)
    if device in vast_id:
        check_output(['vastai', 'destroy', 'instance', str(vast_id[device][2])], args.verbose)
        del vast_id[device]

wandb.log({'Latency and Memory': table})

wandb.finish()
