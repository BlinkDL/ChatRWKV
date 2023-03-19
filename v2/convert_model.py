import os, sys, argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

from rwkv.model import RWKV

VALID_INPUT_FORMATS = set(('pt',))
VALID_OUTPUT_FORMATS = VALID_INPUT_FORMATS

def get_args():
  p = argparse.ArgumentParser(prog = 'convert_model', description = 'Utility for converting RWKV model files')
  p.add_argument('-i', '--in', metavar = 'INPUT', help = 'Filename for input model.', required = True)
  p.add_argument('-o', '--out', metavar = 'OUTPUT', help = 'Filename for output model.', required = True)
  p.add_argument('-s', '--strategy', help = 'See documentation for strategy format definition. Please remember to quote the strategy as it contains spaces and special characters.', required = True)
  p.add_argument('--input-format', default = 'pt', help = 'Input model format. Supported: pt (PyTorch .pth/.pt)')
  p.add_argument('--output-format', default = 'pt', help = 'Output model format. Supported: pt (PyTorch .pth/.pt)')
  p.add_argument('-q', '--quiet', action = 'store_true', help = 'Suppress normal output, only show errors.')
  return p.parse_args()


def load_model(args):
  try:
    model = RWKV(getattr(args, 'in'), args.strategy, use_pinned_memory = False, verbose = not args.quiet)
  except Exception as e:
    print(f'!! Loading model failed: {e}', file = sys.stderr)
    sys.exit(1)
  return model


def save_model(args, model):
  if not args.quiet:
    print(f'** Saving converted model to: {args.out}')
  try:
    model.save_preconverted(args.out)
  except Exception as e:
    print(f'!! Saving model failed: {e}', file = sys.stderr)
    sys.exit(1)


def main():
  args = get_args()
  if not args.quiet:
    print(f'** convert_model running with arguents: {args}')
  if args.input_format not in VALID_INPUT_FORMATS:
    print(f'!! Unsupported input format {args.input_format}, must be one of: {", ".join(VALID_INPUT_FORMATS)}', file = sys.stderr)
    sys.exit(1)
  if args.output_format not in VALID_OUTPUT_FORMATS:
    print(f'!! Unsupported output format {args.output_format}, must be one of: {", ".join(VALID_OUTPUT_FORMATS)}', file = sys.stderr)
    sys.exit(1)
  model = load_model(args)
  save_model(args, model)
  if not args.quiet:
    print('** Successful completion.')


if __name__ == '__main__':
  main()
