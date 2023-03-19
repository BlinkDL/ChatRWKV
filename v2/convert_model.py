import os, sys, argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

from rwkv.model import RWKV

# python convert_model.py --in '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230313-ctx8192-test1050' --out 'fp16_RWKV-4-Pile-14B-20230313-ctx8192-test1050' --strategy 'cuda fp16'
# python convert_model.py --in '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20230109-ctx4096' --out 'fp16_RWKV-4-Pile-7B-20230109-ctx4096' --strategy 'cuda fp16'
# python convert_model.py --in '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096' --out 'fp16i8_and_fp16_RWKV-4-Pile-3B-20221110-ctx4096' --strategy 'cuda fp16i8 *10 -> cuda fp16'

def get_args():
  p = argparse.ArgumentParser(prog = 'convert_model', description = 'Convert RWKV model for faster loading and saves cpu RAM.')
  p.add_argument('--in', metavar = 'INPUT', help = 'Filename for input model.', required = True)
  p.add_argument('--out', metavar = 'OUTPUT', help = 'Filename for output model.', required = True)
  p.add_argument('--strategy', help = 'Please quote the strategy as it contains spaces and special characters. See https://pypi.org/project/rwkv/ for strategy format definition.', required = True)
  p.add_argument('--quiet', action = 'store_true', help = 'Suppress normal output, only show errors.')
  return p.parse_args()

args = get_args()
if not args.quiet:
  print(f'** {args}')
  
RWKV(getattr(args, 'in'), args.strategy, verbose = not args.quiet, convert_and_save_and_exit = args.out)
