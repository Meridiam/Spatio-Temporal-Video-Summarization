import argparse
import json
import os
import re
import time

import imageio
from loguru import logger
from scipy.spatial.transform import Rotation

from sklearn.cluster import DBSCAN
import numpy as np

from utils import parse_framewise_camera_pose, parse_intrinsics


def make_parser():
    parser = argparse.ArgumentParser("Depthmap Generator")
    parser.add_argument(
        "--input_path", default="", help="point cloud file pathname"
    )
    parser.add_argument(
        "--framewise_path", default="", help="framewise camera pose pathname"
    )
    parser.add_argument(
        "--intrinsics_path", default="", help="camera intrinsics file pathname"
    )
    parser.add_argument(
        "--save_path", default="DepthMap_outputs", help="name of save folder"
    )
    return parser


def parse_point_cloud(input_path):
    point_cloud = {}
    with open(input_path, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            toks = re.split(r'\s', line)[1:]
            entries = len(toks) // 5

            for entry_num in range(entries):
                base_idx = 5 * entry_num

                point_id = toks[base_idx]
                conf = toks[base_idx + 1]
                x = toks[base_idx + 2]
                y = toks[base_idx + 3]
                z = toks[base_idx + 4]

                point_cloud[point_id] = {
                    "point": [float(x), float(y), float(z)],
                    "confidence": float(conf)
                }

            logger.info(f'Parsed {entries} entries in line {i}')
            i += 1

    return [v["point"] for k, v in point_cloud.items()]


def denoise_point_cloud(point_cloud):
    orig_count = len(point_cloud)

    # fit clustering over candidate point cloud
    # best result so far: eps=0.2, min_samples=30
    clustering = DBSCAN(eps=0.2, min_samples=30)
    clustering.fit(point_cloud)

    # remove outliers
    mask = clustering.labels_ != -1

    noise_mask = clustering.labels_ == -1

    denoised_point_cloud = point_cloud[mask]

    noise_point_cloud = point_cloud[noise_mask]

    logger.info(f'Culled {orig_count - len(denoised_point_cloud)} out of {orig_count} candidate points')

    return denoised_point_cloud, noise_point_cloud


def apply_inverse_camera_transform(camera_pose, point):
    translated = point - np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])

    rot = Rotation.from_euler('xyz', [camera_pose["roll"], camera_pose["pitch"], camera_pose["yaw"]], degrees=True)

    return rot.inv().apply(translated)


def world_to_image_space(camera_pose, point, fx_d, fy_d, cx_d, cy_d):
    # get 3d pos in camera local coordinate frame
    local_3d_pos = apply_inverse_camera_transform(camera_pose, point)

    image_x = (local_3d_pos[0] * fy_d / local_3d_pos[2]) + cy_d
    image_y = 1280 - ((local_3d_pos[1] * fx_d / local_3d_pos[2]) + cx_d)

    return image_x, image_y, local_3d_pos[2]


def main(args):
    if args.input_path == "":
        print(f'Point cloud path name cannot be blank!')
        exit(1)

    if args.framewise_path == "":
        print(f'Framewise camera pose path name cannot be blank!')
        exit(1)

    if args.intrinsics_path == "":
        print(f'Camera intrinsics path name cannot be blank!')
        exit(1)

    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(save_folder, exist_ok=True)

    depth_savedir = os.path.join(save_folder, 'depth')
    os.makedirs(depth_savedir, exist_ok=True)

    # Parse and denoise ARFoundation point cloud
    point_cloud = np.array(parse_point_cloud(args.input_path))

    with open(os.path.join(save_folder, 'raw_point_cloud'), 'w+') as f:
        f.write(f'{json.dumps(point_cloud.tolist())}')

    point_cloud, noise = denoise_point_cloud(point_cloud)

    with open(os.path.join(save_folder, 'denoised_point_cloud'), 'w+') as f:
        f.write(f'{json.dumps(point_cloud.tolist())}')

    with open(os.path.join(save_folder, 'noise'), 'w+') as f:
        f.write(f'{json.dumps(noise.tolist())}')

    # project point cloud to image space of each frame and calculate binned depth map
    fx_d, fy_d, cx_d, cy_d = parse_intrinsics(args.intrinsics_path)

    camera_poses = parse_framewise_camera_pose(args.framewise_path)

    last_frame = max(camera_poses, key=int)

    for frame, camera_pose in camera_poses.items():
        depthmap = np.full((160, 90), -1.0)

        for point in point_cloud:
            image_x, image_y, depth = world_to_image_space(camera_pose, point, fx_d, fy_d, cx_d, cy_d)

            depthmap_x = int(image_x // 8)
            depthmap_y = int(image_y // 8)

            if depth > 0 and 0 <= depthmap_x < 90 and 0 <= depthmap_y < 160:
                if depthmap[depthmap_y, depthmap_x] < 0 or depth < depthmap[depthmap_y, depthmap_x]:
                    depthmap[depthmap_y, depthmap_x] = depth

        imageio.imwrite(os.path.join(depth_savedir, f'{frame}.exr'), depthmap.astype("float32"))

        logger.info(f'Processed depthmap for frame {frame} / {last_frame}')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
