from argparse import ArgumentParser
import math

parser = ArgumentParser()
parser.add_argument('f_in', help='file path')
parser.add_argument('f_out', help='file path')
parser.add_argument('nchunks', type=int)
parser.add_argument('ichunk', type=int)
parser.add_argument('--mc', action = 'store_true')
args = parser.parse_args()

all_files = [i.strip() for i in open(args.f_in) if i.strip() and not i.startswith('#')]
chunk_size = math.ceil(len(all_files) / args.nchunks)
infiles = all_files[chunk_size*args.ichunk:chunk_size*(args.ichunk+1)]
if infiles:
    print(len(infiles), "to be processed.")
else:
    print('Nothing to be done, no files left')
    exit(0)

