import os, sys, glob, argparse
sys.path.insert(0, 'src')
import data_utils


parser = argparse.ArgumentParser()

parser.add_argument('--dirpaths',
    nargs='+', type=str, required=True, help='Path to directory containing files')
parser.add_argument('--ext',
    default='.png', help='File extensions to fetch')
parser.add_argument('--output_path',
    required=True, help='Path to save file')

args = parser.parse_args()


paths = []

for dirpath in args.dirpaths:
    paths += sorted(glob.glob(os.path.join(dirpath, '*' + args.ext)))

output_dirpath = os.path.dirname(args.output_path)

if not os.path.exists(output_dirpath):
    os.makedirs(output_dirpath)

data_utils.write_paths(args.output_path, paths)
