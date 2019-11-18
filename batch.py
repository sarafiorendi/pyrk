from argparse import ArgumentParser
import math
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('f_in', help='file path')
parser.add_argument('f_out', help='file path', default = 'test.root')
parser.add_argument('nchunks', type=int)
parser.add_argument('ichunk', type=int)
parser.add_argument('--mc', action = 'store_true')
args = parser.parse_args()

all_files = [i.strip() for i in open(args.f_in) if i.strip() and not i.startswith('#')]
infiles = [j for i, j in enumerate(all_files) if (i % args.nchunks) == args.ichunk]
if infiles:
    print(len(infiles), "to be processed.")
else:
    print('Nothing to be done, no files left')
    exit(0)

f_out = args.f_out
