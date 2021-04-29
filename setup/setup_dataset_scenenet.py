import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils
from skimage import morphology as skmorph
from sklearn.cluster import MiniBatchKMeans


N_CLUSTER = 384
N_HEIGHT = 240
N_WIDTH = 320
MIN_POINTS = 360
N_INIT_CORNER = 15000
RANDOM_SEED = 1


parser = argparse.ArgumentParser()

parser.add_argument('--sparse_depth_distro_type',   type=str, default='corner')
parser.add_argument('--sequences_to_process',       nargs='+', type=int, default=[1])
parser.add_argument('--n_points',                   type=int, default=N_CLUSTER)
parser.add_argument('--min_points',                 type=int, default=MIN_POINTS)
parser.add_argument('--n_height',                   type=int, default=N_HEIGHT)
parser.add_argument('--n_width',                    type=int, default=N_WIDTH)


args = parser.parse_args()


def process_frame(inputs):
    '''
    Processes a single frame

    Args:
        inputs : tuple
            image path, ground truth path
    Returns:
        str : output sparse depth path
        str : output validity map path
        str : output semi-dense depth (convex hull of sparse points) path
        str : output ground truth path
    '''

    image_path, ground_truth_path = inputs

    # Load image (for corner detection) to generate validity map
    image = cv2.resize(cv2.imread(image_path), (args.n_width, args.n_height))
    image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if args.sparse_depth_distro_type == 'corner':
        # Run Harris corner detector
        corners = cv2.cornerHarris(image, blockSize=5, ksize=3, k=0.04)

        # Extract specified corner locations:
        corners = corners.ravel()
        corner_locs = np.argsort(corners)[0:N_INIT_CORNER]
        corner_map = np.zeros_like(corners)
        corner_map[corner_locs] = 1

        corner_locs = np.unravel_index(corner_locs, (image.shape[0], image.shape[1]))
        corner_locs = np.transpose(np.array([corner_locs[0], corner_locs[1]]))

        kmeans = MiniBatchKMeans(
            n_clusters=args.n_points,
            max_iter=2,
            n_init=1,
            init_size=None,
            random_state=RANDOM_SEED,
            reassignment_ratio=1e-11)
        kmeans.fit(corner_locs)

        # k-Means means as corners
        corner_locs = kmeans.cluster_centers_.astype(np.uint16)
        validity_map = np.zeros_like(image).astype(np.int16)
        validity_map[corner_locs[:, 0], corner_locs[:, 1]] = 1

    elif args.sparse_depth_distro_type == 'uniform':
        indices = \
            np.array([[h, w] for h in range(args.n_height) for w in range(args.n_width)])

        selected_indices = \
            np.random.permutation(range(args.n_height * args.n_width))[0:args.n_points]
        selected_indices = indices[selected_indices]

        validity_map = np.zeros_like(image).astype(np.int16)
        validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0

    ground_truth = data_utils.load_depth(ground_truth_path, multiplier=1000)

    ground_truth = cv2.resize(
        ground_truth,
        (args.n_width, args.n_height),
        interpolation=cv2.INTER_NEAREST)

    sparse_depth = validity_map * ground_truth
    semi_dense_depth = ground_truth * np.where(skmorph.convex_hull_image(validity_map), 1, 0)

    # Shape check
    error_flag = False

    if np.squeeze(sparse_depth).shape != (args.n_height, args.n_width):
        error_flag = True
        print('FAILED: sparse depth height and width do not match specified values')

    if np.squeeze(semi_dense_depth).shape != (args.n_height, args.n_width):
        error_flag = True
        print('FAILED: semi dense depth height and width do not match specified values')

    # Validity map check
    if not np.array_equal(np.unique(validity_map), np.array([0, 1])):
        error_flag = True
        print('FAILED: validity map contains values other than 0 or 1')

    if validity_map.sum() < args.min_points:
        error_flag = True
        print('FAILED: validity map contains fewer points than miniumum point threshold')

    # Depth value check
    if np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0:
        error_flag = True
        print('FAILED: ground truth value less than 0 or greater than 256')

    if np.sum(np.where(semi_dense_depth > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: valid semi dense depth is less than minimum point threshold', np.sum(np.where(semi_dense_depth > 0.0, 1.0, 0.0)))

    if np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: valid ground truth is less than minimum point threshold', np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)))

    # NaN check
    if np.any(np.isnan(sparse_depth)) or np.any(np.isnan(semi_dense_depth)):
        error_flag = True
        print('FAILED: found NaN in sparse depth or semi dense depth')

    if not error_flag:
        # Generate paths
        derived_ground_truth_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'ground_truth')
        derived_sparse_depth_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'sparse_depth')
        derived_validity_map_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'validity_map')
        derived_semi_dense_depth_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'semi_dense_depth')

        # Write to file
        data_utils.save_validity_map(validity_map, derived_validity_map_path)
        data_utils.save_depth(sparse_depth, derived_sparse_depth_path)
        data_utils.save_depth(semi_dense_depth, derived_semi_dense_depth_path)
        data_utils.save_depth(ground_truth, derived_ground_truth_path)
    else:
        print('Found error in {}'.format(ground_truth_path))
        derived_ground_truth_path = 'error'
        derived_sparse_depth_path = 'error'
        derived_validity_map_path = 'error'
        derived_semi_dense_depth_path = 'error'

    return (derived_sparse_depth_path,
            derived_validity_map_path,
            derived_semi_dense_depth_path,
            derived_ground_truth_path)


'''
Process dataset
'''
SCENENET_ROOT_DIRPATH = os.path.join('data', 'scenenet', 'train')
SCENENET_OUT_DIRPATH = os.path.join(
    'data',
    'scenenet_learning_topology_{}'.format(args.sparse_depth_distro_type),
    'train')

TRAIN_OUTPUT_REF_DIRPATH = os.path.join('training', 'scenenet')

TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_validity_map_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_semi_dense_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))


if not os.path.exists(TRAIN_OUTPUT_REF_DIRPATH):
    os.makedirs(TRAIN_OUTPUT_REF_DIRPATH)

output_sparse_depth_paths = []
output_validity_map_paths = []
output_semi_dense_depth_paths = []
output_ground_truth_paths = []

sequence_base_dirpaths = sorted(glob.glob(os.path.join(SCENENET_ROOT_DIRPATH, '*/')))

# Example file structure: data/scenenet/train/1/1000/
for sequence_base_dirpath in sequence_base_dirpaths:

    seq_id = sequence_base_dirpath.split(os.sep)[-2]

    if -1 not in args.sequences_to_process and not int(seq_id) in args.sequences_to_process:
        # Skip sequence if not in list of sequences to process, -1 to process all
        continue

    output_sequence_sparse_depth_paths = []
    output_sequence_validity_map_paths = []
    output_sequence_semi_dense_depth_paths = []
    output_sequence_ground_truth_paths = []

    # Get sequence directories: 1, 2, ...
    sequence_dirpaths = sorted(glob.glob(os.path.join(sequence_base_dirpath, '*/')))

    for sequence_dirpath in sequence_dirpaths:
        # Fetch image and ground truth paths
        image_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'photo', '*.jpg')))
        ground_truth_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'depth', '*.png')))

        # Make sure there exists matching filenames
        assert(len(image_paths) == len(ground_truth_paths))

        for idx in range(len(image_paths)):
            assert os.path.splitext(os.path.basename(image_paths[idx])[0]) == \
                   os.path.splitext(os.path.basename(ground_truth_paths[idx])[0])

        output_sparse_depth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'sparse_depth'))

        output_validity_map_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'validity_map'))

        output_semi_dense_depth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'semi_dense_depth'))

        output_ground_truth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'ground_truth'))

        output_dirpaths = [
            output_sparse_depth_dirpath,
            output_validity_map_dirpath,
            output_semi_dense_depth_dirpath,
            output_ground_truth_dirpath
        ]

        for output_dirpath in output_dirpaths:
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

        print('Processing sequence={}'.format(sequence_dirpath))

        pool_input = [
            (image_paths[idx], ground_truth_paths[idx])
            for idx in range(len(image_paths))
        ]

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)

            for result in pool_results:
                output_sparse_depth_path, \
                    output_validity_map_path, \
                    output_semi_dense_depth_path, \
                    output_ground_truth_path = result

                found_error = \
                    output_sparse_depth_path == 'error' or \
                    output_validity_map_path == 'error' or \
                    output_semi_dense_depth_path == 'error' or \
                    output_ground_truth_path == 'error'

                if found_error:
                    print('Skipping sample due to error')
                    continue
                else:
                    # Collect filepaths
                    output_sequence_sparse_depth_paths.append(output_sparse_depth_path)
                    output_sequence_validity_map_paths.append(output_validity_map_path)
                    output_sequence_semi_dense_depth_paths.append(output_semi_dense_depth_path)
                    output_sequence_ground_truth_paths.append(output_ground_truth_path)

                    # Do the same for the entire dataset
                    output_sparse_depth_paths.append(output_sparse_depth_path)
                    output_validity_map_paths.append(output_validity_map_path)
                    output_semi_dense_depth_paths.append(output_semi_dense_depth_path)
                    output_ground_truth_paths.append(output_ground_truth_path)

        print('Completed generating {} depth samples for sequence={}'.format(
            len(image_paths), sequence_dirpath))

    output_sparse_depth_filepath = \
        TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    output_validity_map_filepath = \
        TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    output_semi_dense_depth_filepath = \
        TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    output_ground_truth_filepath = \
        TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'

    print('Storing {} sparse depth file paths into: {}'.format(
        len(output_sequence_sparse_depth_paths), output_sparse_depth_filepath))
    data_utils.write_paths(
        output_sparse_depth_filepath, output_sequence_sparse_depth_paths)

    print('Storing {} validity map file paths into: {}'.format(
        len(output_sequence_validity_map_paths), output_validity_map_filepath))
    data_utils.write_paths(
        output_validity_map_filepath, output_sequence_validity_map_paths)

    print('Storing {} semi dense depth file paths into: {}'.format(
        len(output_sequence_semi_dense_depth_paths), output_semi_dense_depth_filepath))
    data_utils.write_paths(
        output_semi_dense_depth_filepath, output_sequence_semi_dense_depth_paths)

    print('Storing {} ground truth file paths into: {}'.format(
        len(output_sequence_ground_truth_paths), output_ground_truth_filepath))
    data_utils.write_paths(
        output_ground_truth_filepath, output_sequence_ground_truth_paths)

print('Storing {} sparse depth file paths into: {}'.format(
    len(output_sparse_depth_paths), TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, output_sparse_depth_paths)

print('Storing {} validity map file paths into: {}'.format(
    len(output_validity_map_paths), TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH, output_validity_map_paths)

print('Storing {} semi dense depth file paths into: {}'.format(
    len(output_semi_dense_depth_paths), TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, output_semi_dense_depth_paths)

print('Storing {} ground truth file paths into: {}'.format(
    len(output_ground_truth_paths), TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH, output_ground_truth_paths)
