import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse, math
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils
from sklearn.cluster import MiniBatchKMeans


N_CLUSTER = 384
MIN_POINTS = 300
N_INIT_CORNER = 15000
RANDOM_SEED = 1


parser = argparse.ArgumentParser()

parser.add_argument('--sparse_depth_distro_type',   type=str, default='corner')
parser.add_argument('--sequences_to_process',       nargs='+', type=int, default=[0, 1])
parser.add_argument('--n_points',                   type=int, default=N_CLUSTER)
parser.add_argument('--min_points',                 type=int, default=MIN_POINTS)

args = parser.parse_args()


SCENENET_ROOT_DIRPATH = os.path.join('data', 'scenenet')
SCENENET_OUT_DIRPATH = os.path.join(
    'data',
    'scenenet_scaffnet_{}'.format(args.sparse_depth_distro_type))

TRAIN_OUTPUT_REF_DIRPATH = os.path.join('training', 'scenenet')
VAL_OUTPUT_REF_DIRPATH = os.path.join('validation', 'scenenet')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_image_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_IMAGES_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_images_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH,
    'scenenet_train_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

VAL_IMAGE_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_image_{}.txt'.format(args.sparse_depth_distro_type))
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_GROUND_TRUTH_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_INTRINSICS_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

VAL_IMAGE_SUBSET_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_image_{}-subset.txt'.format(args.sparse_depth_distro_type))
VAL_SPARSE_DEPTH_SUBSET_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_sparse_depth_{}-subset.txt'.format(args.sparse_depth_distro_type))
VAL_GROUND_TRUTH_SUBSET_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_ground_truth_{}-subset.txt'.format(args.sparse_depth_distro_type))
VAL_INTRINSICS_SUBSET_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH,
    'scenenet_val_intrinsics_{}-subset.txt'.format(args.sparse_depth_distro_type))

VAL_DATA_LIST_FILEPATH = os.path.join('setup', 'data_list_val_scenenet.txt')


TRAIN_INVALID_SPARSE_DEPTH_PATHS = [
    os.path.join('train', '0', '241', 'sparse_depth', '2175'),
    os.path.join('train', '0', '77', 'sparse_depth', '2125'),
    os.path.join('train', '1', '1077', 'sparse_depth', '6275'),
    os.path.join('train', '1', '1263', 'sparse_depth', '5875'),
    os.path.join('train', '1', '1657', 'sparse_depth', '5975'),
    os.path.join('train', '1', '1819', 'sparse_depth', '4675'),
    os.path.join('train', '1', '1967', 'sparse_depth', '4375'),
    os.path.join('train', '2', '2071', 'sparse_depth', '5825'),
    os.path.join('train', '2', '2093', 'sparse_depth', '3450'),
    os.path.join('train', '2', '2543', 'sparse_depth', '7350'),
    os.path.join('train', '2', '2978', 'sparse_depth', '6575'),
    os.path.join('train', '3', '3130', 'sparse_depth', '200'),
    os.path.join('train', '3', '3130', 'sparse_depth', '4625'),
    os.path.join('train', '3', '3195', 'sparse_depth', '1050'),
    os.path.join('train', '3', '3654', 'sparse_depth', '6600'),
    os.path.join('train', '3', '3707', 'sparse_depth', '5425'),
    os.path.join('train', '3', '3815', 'sparse_depth', '4175'),
    os.path.join('train', '4', '4215', 'sparse_depth', '3025'),
    os.path.join('train', '4', '4842', 'sparse_depth', '6275'),
    os.path.join('train', '5', '5139', 'sparse_depth', '4875'),
    os.path.join('train', '5', '5306', 'sparse_depth', '5500'),
    os.path.join('train', '5', '5353', 'sparse_depth', '7125'),
    os.path.join('train', '5', '5553', 'sparse_depth', '5300'),
    os.path.join('train', '5', '5659', 'sparse_depth', '4950'),
    os.path.join('train', '5', '5985', 'sparse_depth', '6225'),
    os.path.join('train', '6', '6212', 'sparse_depth', '5825'),
    os.path.join('train', '6', '6557', 'sparse_depth', '550'),
    os.path.join('train', '6', '6597', 'sparse_depth', '1425'),
    os.path.join('train', '6', '6932', 'sparse_depth', '5925'),
    os.path.join('train', '6', '6972', 'sparse_depth', '5700'),
    os.path.join('train', '7', '7161', 'sparse_depth', '325'),
    os.path.join('train', '7', '7294', 'sparse_depth', '1800'),
    os.path.join('train', '7', '7534', 'sparse_depth', '6675'),
    os.path.join('train', '7', '7774', 'sparse_depth', '4925'),
    os.path.join('train', '8', '8067', 'sparse_depth', '5475'),
    os.path.join('train', '8', '8927', 'sparse_depth', '5400'),
    os.path.join('train', '8', '8976', 'sparse_depth', '6250'),
    os.path.join('train', '8', '8993', 'sparse_depth', '2650'),
    os.path.join('train', '9', '9782', 'sparse_depth', '2575'),
    os.path.join('train', '9', '9782', 'sparse_depth', '5850'),
    os.path.join('train', '9', '9847', 'sparse_depth', '5175'),
    os.path.join('train', '9', '9858', 'sparse_depth', '6825'),
    os.path.join('train', '10', '10138', 'sparse_depth', '6625'),
    os.path.join('train', '10', '10180', 'sparse_depth', '4800'),
    os.path.join('train', '10', '10454', 'sparse_depth', '3800'),
    os.path.join('train', '10', '10777', 'sparse_depth', '425'),
    os.path.join('train', '11', '11032', 'sparse_depth', '2825'),
    os.path.join('train', '11', '11567', 'sparse_depth', '375'),
    os.path.join('train', '11', '11616', 'sparse_depth', '6725'),
    os.path.join('train', '11', '11757', 'sparse_depth', '325'),
    os.path.join('train', '12', '12126', 'sparse_depth', '6925'),
    os.path.join('train', '12', '12169', 'sparse_depth', '975'),
    os.path.join('train', '12', '12295', 'sparse_depth', '1975'),
    os.path.join('train', '12', '12450', 'sparse_depth', '4475'),
    os.path.join('train', '12', '12533', 'sparse_depth', '4275'),
    os.path.join('train', '12', '12547', 'sparse_depth', '1050'),
    os.path.join('train', '13', '13300', 'sparse_depth', '6525'),
    os.path.join('train', '13', '13376', 'sparse_depth', '4350'),
    os.path.join('train', '13', '13422', 'sparse_depth', '5250'),
    os.path.join('train', '13', '13996', 'sparse_depth', '600'),
    os.path.join('train', '13', '13996', 'sparse_depth', '625'),
    os.path.join('train', '13', '13996', 'sparse_depth', '650'),
    os.path.join('train', '14', '14072', 'sparse_depth', '1150'),
    os.path.join('train', '14', '14228', 'sparse_depth', '5250'),
    os.path.join('train', '15', '15292', 'sparse_depth', '7175'),
    os.path.join('train', '15', '15488', 'sparse_depth', '4200')
]

VAL_INVALID_SPARSE_DEPTH_PATHS = [
    os.path.join('val', '0', '728', 'sparse_depth', '2925'),
    os.path.join('val', '0', '929', 'sparse_depth', '3550')
]


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t-1,
            image path at time t,
            image path at time t+1,
            ground truth path at time t
            flag to determine whether to create image triplets
    Returns:
        str : output image path
        str : output images concatenated path
        str : output sparse depth path
        str : output ground truth path
    '''

    image_prev_path, \
        image_curr_path, \
        image_next_path, \
        ground_truth_path, \
        do_create_image_triplets = inputs

    # Load images
    if do_create_image_triplets:
        image_prev = cv2.imread(image_prev_path)
        image_curr = cv2.imread(image_curr_path)
        image_next = cv2.imread(image_next_path)

        images = np.concatenate([image_prev, image_curr, image_next], axis=1)
    else:
        image_curr = cv2.imread(image_curr_path)

    # Convert to gray scale for corner detection to generate validity map
    image_curr_gray = np.float32(cv2.cvtColor(image_curr, cv2.COLOR_BGR2GRAY))

    n_height, n_width = image_curr.shape[:2]

    if args.sparse_depth_distro_type == 'corner':
        # Run Harris corner detector
        corners = cv2.cornerHarris(image_curr_gray, blockSize=5, ksize=3, k=0.04)

        # Extract specified corner locations:
        corners = corners.ravel()
        corner_locs = np.argsort(corners)[0:N_INIT_CORNER]
        corner_map = np.zeros_like(corners)
        corner_map[corner_locs] = 1

        corner_locs = np.unravel_index(corner_locs, (n_height, n_width))
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
        validity_map = np.zeros_like(image_curr_gray).astype(np.int16)
        validity_map[corner_locs[:, 0], corner_locs[:, 1]] = 1

    elif args.sparse_depth_distro_type == 'uniform':
        indices = \
            np.array([[h, w] for h in range(n_height) for w in range(n_width)])

        selected_indices = \
            np.random.permutation(range(n_height * n_width))[0:args.n_points]
        selected_indices = indices[selected_indices]

        validity_map = np.zeros_like(image_curr_gray).astype(np.int16)
        validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0
    elif args.sparse_depth_distro_type == 'dense':
        validity_map = np.ones_like(image_curr_gray).astype(np.uint8)

    ground_truth = data_utils.load_depth(ground_truth_path, multiplier=1000)

    sparse_depth = validity_map * ground_truth

    error_flag = False

    # Depth value check
    if np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0:
        error_flag = True
        print('FAILED: min ground truth value less than 0 or greater than 256')

    if np.sum(np.where(sparse_depth > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: valid sparse depth is less than minimum point threshold', np.sum(np.where(sparse_depth > 0.0, 1.0, 0.0)))

    # NaN check
    if np.any(np.isnan(sparse_depth)):
        error_flag = True
        print('FAILED: found NaN in sparse depth')

    if np.any(np.isnan(ground_truth)):
        error_flag = True
        print('FAILED: found NaN in ground truth')

    if not error_flag:
        # Generate paths
        derived_images_path = image_curr_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
        derived_ground_truth_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'ground_truth')
        derived_sparse_depth_path = ground_truth_path \
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH) \
            .replace('depth', 'sparse_depth')

        # Write to file
        if do_create_image_triplets:
            cv2.imwrite(derived_images_path, images)

        data_utils.save_depth(sparse_depth, derived_sparse_depth_path)
        data_utils.save_depth(ground_truth, derived_ground_truth_path)
    else:
        print('Found error in {}'.format(ground_truth_path))
        derived_images_path = 'error'
        derived_ground_truth_path = 'error'
        derived_sparse_depth_path = 'error'

    if do_create_image_triplets:
        return (image_curr_path,
                derived_images_path,
                derived_sparse_depth_path,
                derived_ground_truth_path)
    else:
        return (image_curr_path,
                derived_sparse_depth_path,
                derived_ground_truth_path)

def get_camera_intrinsic(vfov=45, hfov=60, width=320, height=240):
    '''
    Returns a 3 x 3 camera intrinsics matrix

    Arg(s):
        vfov : float
            field of view in vertical direction
        hfov : float
            field of view in horizontal direction
        width : int
            image width
        height : int
            image height
    Returns:
        numpy[float] : 3 x 3 intrinsics matrix
    '''

    camera_intrinsics = np.zeros([3, 3])
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (width / 2.0) / math.tan(math.radians(hfov / 2.0))
    camera_intrinsics[0, 2] = width / 2.0
    camera_intrinsics[1, 1] = (height / 2.0) / math.tan(math.radians(vfov / 2.0))
    camera_intrinsics[1, 2] = height / 2.0
    return camera_intrinsics


'''
Create intrinsics matrix shared by all images
'''
intrinsics = get_camera_intrinsic()
intrinsics_path = os.path.join(SCENENET_OUT_DIRPATH, 'intrinsics.npy')

os.makedirs(SCENENET_OUT_DIRPATH, exist_ok=True)
np.save(intrinsics_path, intrinsics)


'''
Process training set
'''
if not os.path.exists(TRAIN_OUTPUT_REF_DIRPATH):
    os.makedirs(TRAIN_OUTPUT_REF_DIRPATH)

train_image_paths = []
train_images_paths = []
train_sparse_depth_paths = []
train_ground_truth_paths = []
train_intrinsics_paths = []

sequence_base_dirpaths = sorted(glob.glob(os.path.join(SCENENET_ROOT_DIRPATH, 'train', '*/')))

# Example file structure: data/scenenet/train/1/1000/
for sequence_base_dirpath in sequence_base_dirpaths:

    seq_id = sequence_base_dirpath.split(os.sep)[-2]

    if -1 not in args.sequences_to_process and not int(seq_id) in args.sequences_to_process:
        # Skip sequence if not in list of sequences to process, -1 to process all
        continue

    train_sequence_image_paths = []
    train_sequence_images_paths = []
    train_sequence_sparse_depth_paths = []
    train_sequence_ground_truth_paths = []
    train_sequence_intrinsics_paths = []

    # Get sequence directories: 1, 2, ...
    sequence_dirpaths = sorted(glob.glob(os.path.join(sequence_base_dirpath, '*/')))

    for sequence_dirpath in sequence_dirpaths:
        # Fetch image and ground truth paths
        image_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'photo', '*.jpg')))
        ground_truth_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'depth', '*.png')))

        # Make sure there exists matching filenames
        assert len(image_paths) == len(ground_truth_paths)

        for idx in range(1, len(image_paths) - 1):
            assert os.path.splitext(os.path.basename(image_paths[idx])[0]) == \
                os.path.splitext(os.path.basename(ground_truth_paths[idx])[0])

        train_images_dirpath = os.path.dirname(
            image_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH))

        train_sparse_depth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'sparse_depth'))

        train_ground_truth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'ground_truth'))

        output_dirpaths = [
            train_images_dirpath,
            train_sparse_depth_dirpath,
            train_ground_truth_dirpath
        ]

        for output_dirpath in output_dirpaths:
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

        print('Processing sequence={}'.format(sequence_dirpath))

        pool_input = [
            (image_paths[idx-1], image_paths[idx], image_paths[idx+1], ground_truth_paths[idx], True)
            for idx in range(1, len(image_paths) - 1)
        ]

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)
            n_generated = 0

            for result in pool_results:
                train_image_path, \
                    train_images_path, \
                    train_sparse_depth_path, \
                    train_ground_truth_path = result

                found_error = \
                    train_image_path == 'error' or \
                    train_images_path == 'error' or \
                    train_sparse_depth_path == 'error' or \
                    train_ground_truth_path == 'error'

                for invalid_sparse_depth_path in TRAIN_INVALID_SPARSE_DEPTH_PATHS:
                    # Filter out samples known to be bad
                    if invalid_sparse_depth_path in train_sparse_depth_path:
                        found_error = True
                        break

                if found_error:
                    continue
                else:
                    n_generated += 1

                    # Collect filepaths
                    train_sequence_image_paths.append(train_image_path)
                    train_sequence_images_paths.append(train_images_path)
                    train_sequence_sparse_depth_paths.append(train_sparse_depth_path)
                    train_sequence_ground_truth_paths.append(train_ground_truth_path)
                    train_sequence_intrinsics_paths.append(intrinsics_path)

                    # Do the same for the entire dataset
                    train_image_paths.append(train_image_path)
                    train_images_paths.append(train_images_path)
                    train_sparse_depth_paths.append(train_sparse_depth_path)
                    train_ground_truth_paths.append(train_ground_truth_path)
                    train_intrinsics_paths.append(intrinsics_path)

        print('Completed generating {} depth samples for sequence={}'.format(
            n_generated, sequence_dirpath))

    # Write file paths for each sequence
    train_image_filepath = \
        TRAIN_IMAGE_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    train_images_filepath = \
        TRAIN_IMAGES_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    train_sparse_depth_filepath = \
        TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    train_ground_truth_filepath = \
        TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'
    train_intrinsics_filepath = \
        TRAIN_INTRINSICS_OUTPUT_FILEPATH[:-4] + '-' + seq_id + '.txt'

    print('Storing {} image file paths into: {}'.format(
        len(train_sequence_image_paths), train_image_filepath))
    data_utils.write_paths(
        train_image_filepath, train_sequence_image_paths)

    print('Storing {} images file paths into: {}'.format(
        len(train_sequence_images_paths), train_images_filepath))
    data_utils.write_paths(
        train_images_filepath, train_sequence_images_paths)

    print('Storing {} sparse depth file paths into: {}'.format(
        len(train_sequence_sparse_depth_paths), train_sparse_depth_filepath))
    data_utils.write_paths(
        train_sparse_depth_filepath, train_sequence_sparse_depth_paths)

    print('Storing {} ground truth file paths into: {}'.format(
        len(train_sequence_ground_truth_paths), train_ground_truth_filepath))
    data_utils.write_paths(
        train_ground_truth_filepath, train_sequence_ground_truth_paths)

    print('Storing {} intrinsics file paths into: {}'.format(
        len(train_sequence_intrinsics_paths), train_intrinsics_filepath))
    data_utils.write_paths(
        train_intrinsics_filepath, train_sequence_intrinsics_paths)

# Write file paths for entire dataset
print('Storing {} image file paths into: {}'.format(
    len(train_image_paths), TRAIN_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_paths)

print('Storing {} images file paths into: {}'.format(
    len(train_images_paths), TRAIN_IMAGES_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_OUTPUT_FILEPATH, train_images_paths)

print('Storing {} sparse depth file paths into: {}'.format(
    len(train_sparse_depth_paths), TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, train_sparse_depth_paths)

print('Storing {} ground truth file paths into: {}'.format(
    len(train_ground_truth_paths), TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH, train_ground_truth_paths)

print('Storing {} intrinsics file paths into: {}'.format(
    len(train_intrinsics_paths), TRAIN_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_OUTPUT_FILEPATH, train_intrinsics_paths)


'''
Process validation set
'''
if not os.path.exists(VAL_OUTPUT_REF_DIRPATH):
    os.makedirs(VAL_OUTPUT_REF_DIRPATH)

val_image_paths = []
val_sparse_depth_paths = []
val_ground_truth_paths = []
val_intrinsics_paths = []

sequence_base_dirpaths = sorted(glob.glob(os.path.join(SCENENET_ROOT_DIRPATH, 'val', '*/')))

# Example file structure: data/scenenet/train/1/1000/
for sequence_base_dirpath in sequence_base_dirpaths:

    # Get sequence directories: 1, 2, ...
    sequence_dirpaths = sorted(glob.glob(os.path.join(sequence_base_dirpath, '*/')))

    for sequence_dirpath in sequence_dirpaths:
        # Fetch image and ground truth paths
        image_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'photo', '*.jpg')))
        ground_truth_paths = sorted(glob.glob(os.path.join(sequence_dirpath, 'depth', '*.png')))

        # Make sure there exists matching filenames
        assert len(image_paths) == len(ground_truth_paths)

        for idx in range(len(image_paths)):
            assert os.path.splitext(os.path.basename(image_paths[idx])[0]) == \
                os.path.splitext(os.path.basename(ground_truth_paths[idx])[0])

        val_sparse_depth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'sparse_depth'))

        val_ground_truth_dirpath = os.path.dirname(
            ground_truth_paths[0]
            .replace(SCENENET_ROOT_DIRPATH, SCENENET_OUT_DIRPATH)
            .replace('depth', 'ground_truth'))

        output_dirpaths = [
            val_sparse_depth_dirpath,
            val_ground_truth_dirpath
        ]

        for output_dirpath in output_dirpaths:
            if not os.path.exists(output_dirpath):
                os.makedirs(output_dirpath)

        print('Processing sequence={}'.format(sequence_dirpath))

        pool_input = [
            (image_paths[idx], image_paths[idx], image_paths[idx], ground_truth_paths[idx], False)
            for idx in range(len(image_paths))
        ]

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)
            n_generated = 0

            for result in pool_results:
                val_image_path, \
                    val_sparse_depth_path, \
                    val_ground_truth_path = result

                found_error = \
                    val_image_path == 'error' or \
                    val_sparse_depth_path == 'error' or \
                    val_ground_truth_path == 'error'

                for invalid_sparse_depth_path in VAL_INVALID_SPARSE_DEPTH_PATHS:
                    # Filter out samples known to be bad
                    if invalid_sparse_depth_path in val_sparse_depth_path:
                        found_error = True
                        break

                if found_error:
                    continue
                else:
                    n_generated += 1

                    val_image_paths.append(val_image_path)
                    val_sparse_depth_paths.append(val_sparse_depth_path)
                    val_ground_truth_paths.append(val_ground_truth_path)
                    val_intrinsics_paths.append(intrinsics_path)

        print('Completed generating {} depth samples for sequence={}'.format(
            n_generated, sequence_dirpath))

'''
Select samples from validation data list
'''
val_ref_paths = data_utils.read_paths(VAL_DATA_LIST_FILEPATH)

idx_selected = []

for idx, path in enumerate(val_image_paths):

    path_parts = path.split(os.sep)
    ref_path = path_parts[2:-1]
    ref_filename = os.path.splitext(path_parts[-1])[0]
    ref_path.append(ref_filename)
    ref_path = os.path.join(*ref_path)

    for val_ref_path in val_ref_paths:
        if ref_path == val_ref_path:
            idx_selected.append(idx)
            val_ref_paths.remove(val_ref_path)
            break

# Grab selected samples
val_image_subset_paths = np.array(val_image_paths)[idx_selected]
val_image_subset_paths = val_image_subset_paths.tolist()

val_sparse_depth_subset_paths = np.array(val_sparse_depth_paths)[idx_selected]
val_sparse_depth_subset_paths = val_sparse_depth_subset_paths.tolist()

val_ground_truth_subset_paths = np.array(val_ground_truth_paths)[idx_selected]
val_ground_truth_subset_paths = val_ground_truth_subset_paths.tolist()

val_intrinsics_subset_paths = np.array(val_intrinsics_paths)[idx_selected]
val_intrinsics_subset_paths = val_intrinsics_subset_paths.tolist()

# Write file paths for entire validation dataset
print('Storing {} image file paths into: {}'.format(
    len(val_image_paths), VAL_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_IMAGE_OUTPUT_FILEPATH, val_image_paths)

print('Storing {} sparse depth file paths into: {}'.format(
    len(val_sparse_depth_paths), VAL_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, val_sparse_depth_paths)

print('Storing {} ground truth file paths into: {}'.format(
    len(val_ground_truth_paths), VAL_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_GROUND_TRUTH_OUTPUT_FILEPATH, val_ground_truth_paths)

print('Storing {} intrinsics file paths into: {}'.format(
    len(val_intrinsics_paths), VAL_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_INTRINSICS_OUTPUT_FILEPATH, val_intrinsics_paths)

# Write file paths for validation subset
print('Storing {} image subset file paths into: {}'.format(
    len(val_image_subset_paths), VAL_IMAGE_SUBSET_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_IMAGE_SUBSET_OUTPUT_FILEPATH, val_image_subset_paths)

print('Storing {} sparse depth subset file paths into: {}'.format(
    len(val_sparse_depth_subset_paths), VAL_SPARSE_DEPTH_SUBSET_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_SPARSE_DEPTH_SUBSET_OUTPUT_FILEPATH, val_sparse_depth_subset_paths)

print('Storing {} ground truth subset file paths into: {}'.format(
    len(val_ground_truth_subset_paths), VAL_GROUND_TRUTH_SUBSET_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_GROUND_TRUTH_SUBSET_OUTPUT_FILEPATH, val_ground_truth_subset_paths)

print('Storing {} intrinsics subset file paths into: {}'.format(
    len(val_intrinsics_subset_paths), VAL_INTRINSICS_SUBSET_OUTPUT_FILEPATH))
data_utils.write_paths(
    VAL_INTRINSICS_SUBSET_OUTPUT_FILEPATH, val_intrinsics_subset_paths)
