import os, sys, argparse, shutil
sys.path.insert(0, 'src')
import data_utils


parser = argparse.ArgumentParser()


parser.add_argument('--paths',                  type=str, required=True)
parser.add_argument('--output_dirpath',         type=str, required=True)
parser.add_argument('--anonymize_filenames',    action='store_true')


args = parser.parse_args()

paths = data_utils.read_paths(args.paths)
paths = sorted(paths)

if not os.path.exists(args.output_dirpath):
    os.makedirs(args.output_dirpath)

for idx, path in enumerate(paths):

    filename = os.path.basename(path)

    if args.anonymize_filenames:
        _, ext = os.path.splitext(filename)
        filename = '{:06d}'.format(idx) + ext

    output_path = os.path.join(args.output_dirpath, filename)

    shutil.copy(path, output_path)
