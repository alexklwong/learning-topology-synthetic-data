import os, sys, glob, subprocess, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
import data_utils


def config_plt():
    plt.box(False)
    plt.axis('off')


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_path',              type=str, required=True)
parser.add_argument('--sparse_depth_path',       type=str, required=True)
parser.add_argument('--output_depth_path',       type=str, required=True)
parser.add_argument('--ground_truth_path',       type=str, default='')
parser.add_argument('--visualization_path',      type=str, required=True)
parser.add_argument('--image_ext',               type=str, default='.png')
parser.add_argument('--depth_ext',               type=str, default='.png')

# Visualization
parser.add_argument('--load_image_triplet',      action='store_true')
parser.add_argument('--cmap',                    type=str, default='gist_stern')
parser.add_argument('--vmin',                    type=float, default=0.10)
parser.add_argument('--vmax',                    type=float, default=100.0)

args = parser.parse_args()


cmap = cm.get_cmap(name=args.cmap)
cmap.set_under(color='black')


image_viz_dirpath = os.path.join(args.visualization_path, 'image')
sparse_depth_viz_dirpath = os.path.join(args.visualization_path, 'sparse_depth')
output_depth_viz_dirpath = os.path.join(args.visualization_path, 'output_depth')
figure_viz_dirpath = os.path.join(args.visualization_path, 'figure')

dirpaths = [
    image_viz_dirpath,
    sparse_depth_viz_dirpath,
    output_depth_viz_dirpath,
    figure_viz_dirpath
]

for dirpath in dirpaths:
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


'''
Fetch file paths from input directories
'''
if os.path.isdir(args.image_path):
    image_paths = \
        sorted(glob.glob(os.path.join(args.image_path, '*' + args.image_ext)))
else:
    image_paths = data_utils.read_paths(args.image_path)

if os.path.isdir(args.sparse_depth_path):
    sparse_depth_paths = \
        sorted(glob.glob(os.path.join(args.sparse_depth_path, '*' + args.depth_ext)))
else:
    sparse_depth_paths = data_utils.read_paths(args.sparse_depth_path)

if os.path.isdir(args.output_depth_path):
    output_depth_paths = \
        sorted(glob.glob(os.path.join(args.output_depth_path, '*' + args.depth_ext)))
else:
    output_depth_paths = data_utils.read_paths(args.output_depth_path)

n_sample = len(image_paths)

assert n_sample == len(sparse_depth_paths)
assert n_sample == len(output_depth_paths)

ground_truth_available = True if args.ground_truth_path != '' else False

if ground_truth_available:

    if os.path.isdir(args.ground_truth_path):
        ground_truth_paths = \
            sorted(glob.glob(os.path.join(args.ground_truth_path, '*' + args.depth_ext)))
    else:
        ground_truth_paths = data_utils.read_paths(args.ground_truth_path)

    assert n_sample == len(ground_truth_paths)


'''
Process image, sparse depth and output depth (and groundtruth)
'''
for idx in range(n_sample):

    sys.stdout.write(
        'Processing {}/{} samples...\r'.format(idx + 1, n_sample))
    sys.stdout.flush()

    image_path = image_paths[idx]
    sparse_depth_path = sparse_depth_paths[idx]
    output_depth_path = output_depth_paths[idx]

    # Set up output path
    filename = os.path.basename(output_depth_path)
    figure_viz_path = os.path.join(figure_viz_dirpath, filename)

    # Load image, sparse depth and output depth (and groundtruth)
    image = Image.open(image_paths[idx]).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)

    if args.load_image_triplet:
        image = np.split(image, indices_or_sections=3, axis=1)[1]

    sparse_depth = data_utils.load_depth(sparse_depth_path)

    output_depth = data_utils.load_depth(output_depth_path)

    # Save image, sparse depth and output depth
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    config_plt()
    ax.imshow(image)

    image_viz_path = os.path.join(image_viz_dirpath, filename)

    plt.savefig(image_viz_path)
    plt.close()
    subprocess.call(["convert", "-trim", image_viz_path, image_viz_path])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    config_plt()
    ax.imshow(sparse_depth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)

    sparse_depth_viz_path = os.path.join(sparse_depth_viz_dirpath, filename)

    plt.savefig(sparse_depth_viz_path)
    plt.close()
    subprocess.call(["convert", "-trim", sparse_depth_viz_path, sparse_depth_viz_path])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    config_plt()
    ax.imshow(output_depth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)

    output_depth_viz_path = os.path.join(output_depth_viz_dirpath, filename)

    plt.savefig(output_depth_viz_path)
    plt.close()
    subprocess.call(["convert", "-trim", output_depth_viz_path, output_depth_viz_path])

    # Set number of rows in output visualization
    n_row = 3

    if ground_truth_available:
        ground_truth_path = ground_truth_paths[idx]
        ground_truth = data_utils.load_depth(ground_truth_path)
        n_row = 5

    # Create figure and grid
    plt.figure(figsize=(75, 25), dpi=40, facecolor='w', edgecolor='k')

    gs = gridspec.GridSpec(n_row, 1, wspace=0.0, hspace=0.0)

    # Plot image, sparse depth, output depth
    ax = plt.subplot(gs[0, 0])
    config_plt()
    ax.imshow(image)

    ax = plt.subplot(gs[1, 0])
    config_plt()
    ax.imshow(sparse_depth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)

    ax = plt.subplot(gs[2, 0])
    config_plt()
    ax.imshow(output_depth, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    # Plot groundtruth if available
    if ground_truth_available:
        error_depth = np.where(
            ground_truth > 0,
            np.abs(output_depth - ground_truth) / ground_truth,
            0.0)

        ax = plt.subplot(gs[3, 0])
        config_plt()
        ax.imshow(error_depth, vmin=0.00, vmax=0.20, cmap='hot')

        ax = plt.subplot(gs[4, 0])
        config_plt()
        ax.imshow(ground_truth, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    plt.savefig(figure_viz_path)
    plt.close()
    subprocess.call(["convert", "-trim", figure_viz_path, figure_viz_path])
