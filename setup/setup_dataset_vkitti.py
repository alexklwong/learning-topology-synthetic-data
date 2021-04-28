import sys, os, glob
import numpy as np
import cv2
import multiprocessing as mp
from skimage import morphology as skmorph
sys.path.insert(0, 'src')
import data_utils


'''
Paths for KITTI dataset
'''
KITTI_ROOT_DIRPATH = os.path.join('data', 'kitti_depth_completion')
KITTI_TRAIN_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_ROOT_DIRPATH, 'train_val_split', 'sparse_depth', 'train')

# To be concatenated to sequence path
KITTI_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')

'''
Paths for Virtual KITTI dataset
'''
VKITTI_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti')
VKITTI_TRAIN_DEPTH_REFPATH = 'vkitti_1.3.1_depthgt'

# Note: we only need to use the clone directory since lighting change only affects RGB
VKITTI_TRAIN_DENSE_DEPTH_DIRPATH = \
    os.path.join(VKITTI_ROOT_DIRPATH, VKITTI_TRAIN_DEPTH_REFPATH)

'''
Output directory
'''
OUTPUT_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti_learning_topology')
OUTPUT_REF_DIRPATH = os.path.join('training', 'vkitti')

OUTPUT_SPARSE_DEPTH_FILEPATH = os.path.join(
    OUTPUT_REF_DIRPATH, 'vkitti_train_sparse_depth.txt')
OUTPUT_VALIDITY_MAP_FILEPATH = os.path.join(
    OUTPUT_REF_DIRPATH, 'vkitti_train_validity_map.txt')
OUTPUT_SEMI_DENSE_DEPTH_FILEPATH = os.path.join(
    OUTPUT_REF_DIRPATH, 'vkitti_train_semi_dense_depth.txt')
OUTPUT_DENSE_DEPTH_FILEPATH = os.path.join(
    OUTPUT_REF_DIRPATH, 'vkitti_train_dense_depth.txt')
OUTPUT_GROUND_TRUTH_FILEPATH = os.path.join(
    OUTPUT_REF_DIRPATH, 'vkitti_train_ground_truth.txt')


def process_frame(inputs):
    '''
    Processes a single depth frame

    Args:
        inputs : tuple
            KITTI sparse depth path,
            Virtual KITTI ground truth path,
            output directory paths in order of:
                sparse depth, validity map, semi-dense depth, dense depth, groundtruth
    Returns:
        str : Virtual KITTI output sparse depth path
        str : Virtual KITTI output validity map path
        str : Virtual KITTI output semi-dense depth (convex hull of sparse points) path
        str : Virtual KITTI output dense depth path (ground truth without sky)
        str : Virtual KITTI output ground truth path
    '''

    # Separate arguments into individual variables
    kitti_sparse_depth_path, vkitti_ground_truth_path, output_dirpaths = inputs

    # Extract validity map from KITTI sparse depth
    _, kitti_validity_map = data_utils.load_depth_with_validity_map(kitti_sparse_depth_path)

    # Load Virtual KITTI ground truth
    vkitti_ground_truth = \
        cv2.imread(vkitti_ground_truth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Convert Virtual KITTI ground truth to meters
    vkitti_ground_truth = vkitti_ground_truth / 100.0

    if kitti_validity_map.shape != vkitti_ground_truth.shape:
        # Resize KITTI validity map to VKITTI size
        kitti_validity_map = cv2.resize(
            kitti_validity_map,
            dsize=(vkitti_ground_truth.shape[1], vkitti_ground_truth.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        assert(np.all(np.unique(kitti_validity_map) == [0, 1]))

    # Get Virtual KITTI dense depth without sky
    vkitti_validity_map = np.ones(vkitti_ground_truth.shape)
    vkitti_validity_map[vkitti_ground_truth > 600.0] = 0.0
    vkitti_dense_depth = vkitti_validity_map * vkitti_ground_truth

    # Get Virtual KITTI sparse depth
    vkitti_sparse_depth = kitti_validity_map * vkitti_dense_depth

    # Get Virtual KITTI semi-dense depth (convex hull of sparse points)
    vkitti_semi_dense_depth = \
        np.where(skmorph.convex_hull_image(kitti_validity_map), 1, 0) * vkitti_dense_depth

    # Create output filepaths
    filename = os.path.basename(vkitti_ground_truth_path)

    output_sparse_depth_dirpath, \
        output_validity_map_dirpath, \
        output_semi_dense_depth_dirpath, \
        output_dense_depth_dirpath, \
        output_ground_truth_dirpath = output_dirpaths

    output_sparse_depth_path = os.path.join(output_sparse_depth_dirpath, filename)
    output_validity_map_path = os.path.join(output_validity_map_dirpath, filename)
    output_semi_dense_depth_path = os.path.join(output_semi_dense_depth_dirpath, filename)
    output_dense_depth_path = os.path.join(output_dense_depth_dirpath, filename)
    output_ground_truth_path = os.path.join(output_ground_truth_dirpath, filename)

    # Write to disk
    data_utils.save_depth(vkitti_sparse_depth, output_sparse_depth_path)
    data_utils.save_validity_map(kitti_validity_map, output_validity_map_path)
    data_utils.save_depth(vkitti_semi_dense_depth, output_semi_dense_depth_path)
    data_utils.save_depth(vkitti_dense_depth, output_dense_depth_path)
    data_utils.save_depth(vkitti_ground_truth, output_ground_truth_path)

    return (output_sparse_depth_path,
            output_validity_map_path,
            output_semi_dense_depth_path,
            output_dense_depth_path,
            output_ground_truth_path)


'''
Select KITTI and Virtual KITTI paths
'''
# Obtain the set of sequence dirpaths
kitti_sequence_dirpaths = glob.glob(os.path.join(KITTI_TRAIN_SPARSE_DEPTH_DIRPATH, '*/'))
vkitti_sequence_dirpaths = glob.glob(os.path.join(VKITTI_TRAIN_DENSE_DEPTH_DIRPATH, '*/'))

# Get the longest sequence from VKITTI
max_vkitti_filepaths = 0
for vkitti_sequence_dirpath in vkitti_sequence_dirpaths:
    # Select filepaths in Virtual KITTI sequence
    vkitti_sequence_dirpath = os.path.join(vkitti_sequence_dirpath, 'clone')
    vkitti_sequence_filepaths = glob.glob(os.path.join(vkitti_sequence_dirpath, '*.png'))
    n_vkitti_filepaths = len(vkitti_sequence_filepaths)

    if n_vkitti_filepaths > max_vkitti_filepaths:
        max_vkitti_filepaths = n_vkitti_filepaths

# Select from the KITTI sequences that have at least the number of files as VKITTI
kitti_sequence_dirpath_pool = []
for kitti_sequence_dirpath in kitti_sequence_dirpaths:
    # Select filepaths in KITTI sequence
    kitti_sequence_filepaths = glob.glob(
        os.path.join(kitti_sequence_dirpath, KITTI_SPARSE_DEPTH_REFPATH, 'image_02', '*.png'))
    n_kitti_filepaths = len(kitti_sequence_filepaths)

    if n_kitti_filepaths >= max_vkitti_filepaths:
        kitti_sequence_dirpath_pool.append(kitti_sequence_dirpath)


'''
Process data to generate sparse depth for Virtual KITTI
'''
if not os.path.exists(OUTPUT_REF_DIRPATH):
    os.makedirs(OUTPUT_REF_DIRPATH)

output_sparse_depth_paths = []
output_validity_map_paths = []
output_semi_dense_depth_paths = []
output_dense_depth_paths = []
output_ground_truth_paths = []
for vkitti_sequence_dirpath in vkitti_sequence_dirpaths:
    print('Processing Virtual KITTI sequence: {}'.format(vkitti_sequence_dirpath))

    # Select filepath in Virtual KITTI sequence
    vkitti_sequence_dirpath = os.path.join(vkitti_sequence_dirpath, 'clone')
    vkitti_sequence = vkitti_sequence_dirpath.split(os.sep)[-2]
    vkitti_sequence_depth_filepaths = sorted(glob.glob(os.path.join(vkitti_sequence_dirpath, '*.png')))
    n_vkitti_filepaths = len(vkitti_sequence_depth_filepaths)

    output_sequence_dirpath = os.path.join(
        OUTPUT_ROOT_DIRPATH, VKITTI_TRAIN_DEPTH_REFPATH, vkitti_sequence)

    for kitti_sequence_dirpath in kitti_sequence_dirpath_pool:
        # Select KITTI sequence, since it is a directory last element is empty so grab the second til last
        kitti_sequence = kitti_sequence_dirpath.split(os.sep)[-2]
        kitti_sequence_dirpath = os.path.join(kitti_sequence_dirpath, KITTI_SPARSE_DEPTH_REFPATH)

        for camera_dirpath in ['image_02', 'image_03']:
            kitti_sequence_filepaths = sorted(glob.glob(
                os.path.join(kitti_sequence_dirpath, camera_dirpath, '*.png')))
            kitti_sequence_filepaths = kitti_sequence_filepaths[0:n_vkitti_filepaths]

            output_sparse_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, camera_dirpath, 'sparse')
            output_validity_map_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, camera_dirpath, 'validity_map')
            output_semi_dense_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, camera_dirpath, 'semi_dense')
            output_dense_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, camera_dirpath, 'dense')
            output_ground_truth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, camera_dirpath, 'ground_truth')

            output_dirpaths = [
                output_sparse_depth_dirpath,
                output_validity_map_dirpath,
                output_semi_dense_depth_dirpath,
                output_dense_depth_dirpath,
                output_ground_truth_dirpath
            ]

            for output_dirpath in output_dirpaths:
                if not os.path.exists(output_dirpath):
                    os.makedirs(output_dirpath)

            pool_input = [
                (kitti_sequence_filepaths[idx], vkitti_sequence_depth_filepaths[idx], output_dirpaths)
                for idx in range(n_vkitti_filepaths)
            ]

            with mp.Pool() as pool:
                pool_results = pool.map(process_frame, pool_input)

                for result in pool_results:
                    output_sparse_depth_path, \
                        output_validity_map_path, \
                        output_semi_dense_depth_path, \
                        output_dense_depth_path, \
                        output_ground_truth_path = result

                    # Collect filepaths
                    output_sparse_depth_paths.append(output_sparse_depth_path)
                    output_validity_map_paths.append(output_validity_map_path)
                    output_semi_dense_depth_paths.append(output_semi_dense_depth_path)
                    output_dense_depth_paths.append(output_dense_depth_path)
                    output_ground_truth_paths.append(output_ground_truth_path)

            print('Completed generating {} depth samples for using KITTI sequence={} camera={}'.format(
                n_vkitti_filepaths, kitti_sequence, camera_dirpath))

print('Storing sparse depth file paths into: %s' % OUTPUT_SPARSE_DEPTH_FILEPATH)
data_utils.write_paths(
    OUTPUT_SPARSE_DEPTH_FILEPATH, output_sparse_depth_paths)

print('Storing validity map file paths into: %s' % OUTPUT_VALIDITY_MAP_FILEPATH)
data_utils.write_paths(
    OUTPUT_VALIDITY_MAP_FILEPATH, output_validity_map_paths)

print('Storing semi dense depth file paths into: %s' % OUTPUT_SEMI_DENSE_DEPTH_FILEPATH)
data_utils.write_paths(
    OUTPUT_SEMI_DENSE_DEPTH_FILEPATH, output_semi_dense_depth_paths)

print('Storing dense depth file paths into: %s' % OUTPUT_DENSE_DEPTH_FILEPATH)
data_utils.write_paths(
    OUTPUT_DENSE_DEPTH_FILEPATH, output_dense_depth_paths)

print('Storing ground-truth depth file paths into: %s' % OUTPUT_GROUND_TRUTH_FILEPATH)
data_utils.write_paths(
    OUTPUT_GROUND_TRUTH_FILEPATH, output_ground_truth_paths)
