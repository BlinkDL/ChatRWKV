import gc
import importlib
import io
import pickle
import random
import subprocess
import sys,os

import torch

python = sys.executable

            #   "http://pypi.sdutlinux.org/",
            #   "http://pypi.hustunique.com/",
pip_mirror = [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
              "http://mirrors.aliyun.com/pypi/simple/",
              "https://pypi.mirrors.ustc.edu.cn/simple/",
              "http://pypi.douban.com/simple/"
              ]

def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def run_pip(args, desc=None):
    lan = os.getenv('LANG')
    if 'zh_CN' in lan:
        index_url = random.choice(pip_mirror)
    else:
        index_url=None
    if 'https' in index_url:
        trust = None
    else:
        host_url = index_url.removeprefix("http://")
        host_url = host_url.split('/')[0]
        trust = '--trusted-host ' + host_url
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    print(run(f'"{python}" -m pip {args} --prefer-binary{index_url_line} {trust}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}"))

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None

def ReadModelStrategy(model_path):
    w = torch.load(model_path)
    if '_strategy' in w:
        stgy = w['_strategy']
    else:
        stgy = 'cuda fp16'
    if '_version' in w:
        ver = w['_version']
    else:
        ver = 0.7
    return stgy, ver

def ReadMemoryInfo():
    device_list = {}
    if not is_installed("psutil"):
        try:
            run_pip(f"install psutil", "psutil")
            import psutil
        except ModuleNotFoundError:
            print("install psutil failed")
            return 0, device_list
    else:
        import psutil
        print("psutil is already installed")

    # 获取系统内存信息
    mem = psutil.virtual_memory()
    # 获取可用内存大小（单位为字节）
    available_mem = mem.available
    # 将字节转换为MB
    available_mem_gb = available_mem / 1024 / 1024 / 1024
    print("系统剩余内存大小为：{:.2f} GB".format(available_mem_gb))

    import pynvml
    # 初始化 pynvml 库
    pynvml.nvmlInit()
    # 获取 GPU 设备数量
    device_count = pynvml.nvmlDeviceGetCount()
    # 遍历所有 GPU 设备，打印其显存大小
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print("设备 {} 剩余显存大小为: {:.2f} GB, 总显存大小为：{:.2f} GB".format(i, info.free / (1024 ** 3), info.total / (1024 ** 3)))
        if info.free * 2 < info.total:
            print("显存使用过多，请确定是否继续")
        device_list[i] = info.free / (1024 ** 3)
    # 释放 pynvml 库资源
    pynvml.nvmlShutdown()

    # import torch
    # device = torch.device("cuda")
    # print("可用显存大小为：{:.2f} GB".format(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024))
    return available_mem_gb, device_list

def CheckStrategy(model_path):
    memory, device_list = ReadMemoryInfo()
    filesize = os.path.getsize(model_path)
    filesize_gb = filesize / 1024 / 1024 / 1024
    if filesize_gb > memory:
        lan = os.getenv('LANG')
        if 'zh_CN' in lan:
            print("模型无法加载，系统可用内存{:.2f} GB小于模型大小{:.2f} GB，请增加虚拟内存".format(memory, filesize_gb))
        else:
            print("cannot load model, Available memory{:.2f} GB is less than file size{:.2f} GB, increase visual memory".format(memory, filesize_gb))
        return ''

    try:
        strategy, version = ReadModelStrategy(model_path)
    except ModuleNotFoundError:
        print("load model failed")
        return ''

    gc.collect()

    if (strategy != 'cuda fp16') and (strategy != 'cuda fp16i8'):
        print(f"strategy was Customized by yourself, Return as it is")
        return strategy

    fp16_layersize = 0.415
    int8_layersize = 0.224
    if device_list.__len__() == 1:
        gpu_memory = device_list[0] - 1
        if strategy == 'cuda fp16':
            if gpu_memory > filesize_gb:
                return strategy
            elif gpu_memory > filesize_gb / 2:
                total_layer = filesize_gb / fp16_layersize
                fpi8_layer = (filesize_gb - gpu_memory) / (fp16_layersize - int8_layersize)
                if fpi8_layer > total_layer:
                    return f'cuda fp16i8'
                return f'cuda fp16i8 *{int(fpi8_layer)} -> cuda fp16'
            else:
                layer = gpu_memory / int8_layersize
                return f'cuda fp16i8 *{int(layer)}+'
        else:
            if gpu_memory > filesize_gb:
                return strategy
            else:
                layer = gpu_memory / int8_layersize
                if layer == 0:
                    return ''
                return f'cuda fp16i8 *{int(layer)}+'
    else:
        sort_devices = sorted(device_list.items(), key=lambda x: x[1], reverse=True)
        mode = strategy.removeprefix('cuda ')
        left_gpu_memory = 0
        for value in sort_devices:
            left_gpu_memory += value[1]
        if left_gpu_memory < filesize_gb:
            print("can't support your model file, total gpu memory is {:.2f} GB, but your model file size is {:.2f} GB".format(left_gpu_memory, filesize_gb))
            return ''

        if strategy == 'cuda fp16':
            layer_size = fp16_layersize
        else:
            layer_size = int8_layersize
        out_strategy = ''
        for item in sort_devices:
            key = item[0]
            value = item[1]
            if out_strategy.__len__() > 0:
                out_strategy += ' -> '
            
            layer = value / layer_size
            if layer < 1:
                layer = 1
            
            left_gpu_memory -= value
            if left_gpu_memory < 0.1:
                out_strategy += f'cuda:{key} {mode}'
            else:
                out_strategy += f'cuda:{key} {mode} *{int(layer)}'
        return out_strategy

if __name__ == "__main__":
    # strategy = CheckStrategy('F:/RWKVModel/fp16i8_RWKV-4b-Pile-171M-20230202-7922.pth')
    strategy = CheckStrategy('D:/AI/RWKV-4b-Pile-171M-20230202-7922.pth')
    if strategy != '':
        lan = os.getenv('LANG')
        if 'zh_CN' in lan:
            print(f"你的策略: {strategy}")
        else:
            print(f"your strategy is: {strategy}")
