import os, sys, glob, struct, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', required=True, help='Path to file containing paths to images')
parser.add_argument('--depth_path', required=True, help='Path to file containing paths to depth maps')
parser.add_argument('--intrinsics_path', required=True, help='Path to file containing paths to intrinsics')
parser.add_argument('--backproject_and_color', action='store_true')
parser.add_argument('--image_triplet', action='store_true')
parser.add_argument('--output_dirpath', required=True, help='Path to directory to save the point cloud files')

args = parser.parse_args()


def backproject_to_camera(depth, intrinsics, image=None):
    '''
    Backproject image and depth to colored point cloud

    Args:
        depth : numpy
            H x W x 1 array
        intrinsics : numpy
            3 x 3 intrinsics matrix
        image : numpy
            H x W x 3 array
    Returns:
        numpy : N x 6 backprojected and colored points if image is not None else
            N x 3 backprojected points
    '''

    if image is not None:
        assert image.shape[0:2] == depth.shape[0:2]

    n_height, n_width = depth.shape[0:2]

    x = np.linspace(start=0.0, stop=n_width-1.0, num=n_width)
    y = np.linspace(start=0.0, stop=n_height-1.0, num=n_height)

    # Create H x W grids
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')

    # Create H x W x 1 grid (x, y, 1)
    xy_h = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=0)

    # Reshape pixel coordinates to 3 x (H x W)
    xy_h = np.reshape(xy_h, (3, -1))

    # Reshape depth as (H x W)
    depth = np.reshape(depth, -1)

    # Remove any points behind the camera to get N points
    valid_idx = np.where(depth >= 0)
    depth = np.reshape(depth[valid_idx], (1, -1))
    xy_h = np.reshape(xy_h[:, valid_idx], (3, -1))

    # K^-1 [x, y, 1] z
    xyz = np.matmul(np.linalg.inv(intrinsics), xy_h * depth)

    if image is not None:
        # Reshape image to 3 x (H x W)
        rgb = np.reshape(np.transpose(image, (2, 0, 1)), (3, -1))
        rgb = np.reshape(rgb[:, valid_idx], (3, -1))

        # Stack XYZ and RGB together as 6 x N
        xyz_rgb = np.concatenate([xyz, rgb], axis=0)

        return xyz_rgb
    else:
        return xyz

def write_point_cloud(path, point_cloud):
    '''
    Writes point cloud to file

    Args:
        path : str
            filepath to write
        point_cloud : numpy
            N x 3 (x, y, z) or N x 6 (x, y, z, r, g, b) array
    '''

    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[0] == 3 or point_cloud.shape[0] == 6

    if point_cloud.shape[0] == 3:
        xyz = point_cloud
        rgb = int(255 / 2) * np.ones(point_cloud.shape, dtype=np.uint8)
    else:
        xyz = point_cloud[0:3, :]
        rgb = point_cloud[3:6, :].astype(np.uint8)


    with open(path, 'wb') as f:

        n_point = point_cloud.shape[1]

        # Write header of .ply file
        f.write(bytes('ply\n', 'utf-8'))
        f.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        f.write(bytes('element vertex %d\n' % n_point, 'utf-8'))
        f.write(bytes('property float x\n', 'utf-8'))
        f.write(bytes('property float y\n', 'utf-8'))
        f.write(bytes('property float z\n', 'utf-8'))
        f.write(bytes('property uchar red\n', 'utf-8'))
        f.write(bytes('property uchar green\n', 'utf-8'))
        f.write(bytes('property uchar blue\n', 'utf-8'))
        f.write(bytes('end_header\n', 'utf-8'))

        # Write point cloud to .ply file
        for n in range(n_point):
            f.write(bytearray(struct.pack(
                'fffccc',
                xyz[0, n],
                xyz[1, n],
                xyz[2, n],
                rgb[0, n].tostring(),
                rgb[1, n].tostring(),
                rgb[2, n].tostring())))

if __name__ == '__main__':

    # Read image, depth and intrinsics paths
    if os.path.isdir(args.depth_path):
        depth_paths = sorted(glob.glob(os.path.join(args.depth_path, '*.png')))
    else:
        depth_paths = data_utils.read_paths(args.depth_path)

    n_sample = len(depth_paths)

    intrinsics_paths = data_utils.read_paths(args.intrinsics_path)
 
    if args.image_path != '':
        image_paths = data_utils.read_paths(args.image_path)
    else:
        image_paths = [None] * n_sample

    assert n_sample == len(image_paths)
    assert n_sample == len(intrinsics_paths)

    if not os.path.exists(args.output_dirpath):
        os.makedirs(args.output_dirpath)

    for idx in range(len(depth_paths)):

        depth_path = depth_paths[idx]

        filename_depth = os.path.basename(depth_path)

        found_image_path = False
        for image_path in image_paths:

            if filename_depth in image_path:
                found_image_path = True
                break

        assert found_image_path

        sequence_dirpath = image_path.split(os.sep)[-3]

        found_intrinsics_path = False
        for intrinsics_path in intrinsics_paths:

            if sequence_dirpath in intrinsics_path:
                found_intrinsics_path = True
                break

        assert found_intrinsics_path

        # Load image, depth and intrinsics from file
        if args.backproject_and_color:
            image = np.asarray(Image.open(image_path).convert('RGB'), np.uint8)
        else:
            image = None

        if image is not None and args.image_triplet:
            image = np.split(image, indices_or_sections=3, axis=1)[1]

        depth = data_utils.load_depth(depth_path)
        intrinsics = np.load(intrinsics_path)

        # Create point cloud
        point_cloud = backproject_to_camera(depth, intrinsics, image)

        filename = os.path.basename(depth_path)
        output_path = os.path.join(
            args.output_dirpath,
            os.path.splitext(filename)[0] + '.ply')

        write_point_cloud(output_path, point_cloud)

        print('Processed {}/{} samples'.format(idx, n_sample), end='\r')
