import numpy as np
import torch.utils.data
import data_utils


def load_image_triplet(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t - 1
        numpy[float32] : image at t
        numpy[float32] : image at t + 1
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)

    return image1, image0, image2

def load_depth(depth_path, data_format='CHW'):
    '''
    Load depth

    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
    Return:
        numpy[float32] : depth map (1 x H x W)
    '''

    return data_utils.load_depth(depth_path, data_format=data_format)

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs


class ScaffNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) sparse depth
        (2) target dense (ground truth) depth

    Arg(s):
        sparse_depth_paths : list[str]
            paths to sparse depth
        ground_truth_paths : list[str]
            paths to ground truth depth
        cap_dataset_depth_method : str
            remove, set_to_max
        min_dataset_depth : int
            minimum depth to load, any values less will be set to 0.0
        max_dataset_depth : float
            maximum depth to load, any values more will be set to 0.0
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 sparse_depth_paths,
                 ground_truth_paths,
                 cap_dataset_depth_method='set_to_max',
                 min_dataset_depth=-1.0,
                 max_dataset_depth=-1.0,
                 random_crop_shape=None,
                 random_crop_type=None):

        self.sparse_depth_paths = sparse_depth_paths
        self.ground_truth_paths = ground_truth_paths

        self.n_sample = len(self.sparse_depth_paths)
        assert self.n_sample == len(self.ground_truth_paths)

        self.cap_dataset_depth_method = cap_dataset_depth_method
        self.min_dataset_depth = min_dataset_depth
        self.max_dataset_depth = max_dataset_depth

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)

        if self.do_random_crop:
            [sparse_depth, ground_truth] = random_crop(
                inputs=[sparse_depth, ground_truth],
                shape=self.random_crop_shape,
                crop_type=self.random_crop_type)

        if self.min_dataset_depth > 0.0:
            sparse_depth[sparse_depth < self.min_dataset_depth] = 0.0
            ground_truth[ground_truth < self.min_dataset_depth] = 0.0

        if self.max_dataset_depth > 0.0:
            if self.cap_dataset_depth_method == 'remove':
                sparse_depth[sparse_depth > self.max_dataset_depth] = 0.0
                ground_truth[ground_truth > self.max_dataset_depth] = 0.0
            elif self.cap_dataset_depth_method == 'set_to_max':
                sparse_depth[sparse_depth > self.max_dataset_depth] = self.max_dataset_depth
                ground_truth[ground_truth > self.max_dataset_depth] = self.max_dataset_depth

        sparse_depth, ground_truth = [
            T.astype(np.float32)
            for T in [sparse_depth, ground_truth]
        ]

        return sparse_depth, ground_truth

    def __len__(self):
        return self.n_sample

class ScaffNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching sparse depth and ground truth

    Arg(s):
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        ground_truth_paths : list[str]
            paths to ground truth
    '''

    def __init__(self, sparse_depth_paths, ground_truth_paths=None):

        self.sparse_depth_paths = sparse_depth_paths

        self.n_sample = len(self.sparse_depth_paths)

        input_paths = [sparse_depth_paths]

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths
            input_paths.append(ground_truth_paths)

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        inputs = [sparse_depth]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return sparse_depth, and if available, ground_truth
        return inputs

    def __len__(self):
        return len(self.sparse_depth_paths)


class FusionNetStandaloneTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth at time t
        (3) camera intrinsics matrix

    Arg(s):
        images_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=['none']):

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.n_sample = len(images_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            self.random_crop_shape is not None and all([x > 0 for x in self.random_crop_shape])

        # Augmentation
        self.random_crop_type = random_crop_type

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load image at t-1, t, t+1
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth at time t
        sparse_depth0 = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        inputs = [
            image0, image1, image2, sparse_depth0
        ]

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        inputs = inputs + [intrinsics]

        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample


class FusionNetStandaloneInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) ground truth

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        input_paths = [image_paths, sparse_depth_paths]

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths
            input_paths.append(ground_truth_paths)

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_image_triplet(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        inputs = [
            image,
            sparse_depth
        ]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, and if available, ground_truth
        return inputs

    def __len__(self):
        return self.n_sample
