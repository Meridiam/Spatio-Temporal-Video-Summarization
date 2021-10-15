import argparse
import json
import os
import re
import time
from loguru import logger
from scipy.spatial.transform import Rotation

from sklearn.cluster import DBSCAN
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("Depthmap Generator")
    parser.add_argument(
        "--input_path", default="", help="point cloud file pathname"
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
    clustering = DBSCAN(eps=0.1, min_samples=5)
    clustering.fit(point_cloud)

    # remove outliers
    mask = clustering.labels_ != -1

    noise_mask = clustering.labels_ == -1

    denoised_point_cloud = point_cloud[mask]

    noise_point_cloud = point_cloud[noise_mask]

    logger.info(f'Culled {orig_count - len(denoised_point_cloud)} out of {orig_count} candidate points')

    return denoised_point_cloud, noise_point_cloud


def apply_inverse_camera_transform(camera_pose, point):
    point -= np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])

    rot = Rotation.from_euler('xyz', [camera_pose["roll"], camera_pose["pitch"], camera_pose["yaw"]], degrees=True)

    return rot.inv().apply(point)


def world_to_image_space(camera_pose, point):
    # get 3d pos in camera local coordinate frame
    local_3d_pos = apply_inverse_camera_transform(camera_pose, point)

    # TODO: perform local-to-image transform (remember intrinsic X = image Y since intrinsics were
    #   based on rotated image before fix_img is applied)


def main(args):
    if args.input_path == "":
        print(f'Point cloud path name cannot be blank!')
        exit(1)

    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(save_folder, exist_ok=True)

    point_cloud = np.array(parse_point_cloud(args.input_path))

    with open(os.path.join(save_folder, 'raw_point_cloud'), 'w+') as f:
        f.write(f'{json.dumps(point_cloud.tolist())}')

    point_cloud, noise = denoise_point_cloud(point_cloud)

    with open(os.path.join(save_folder, 'denoised_point_cloud'), 'w+') as f:
        f.write(f'{json.dumps(point_cloud.tolist())}')

    with open(os.path.join(save_folder, 'noise'), 'w+') as f:
        f.write(f'{json.dumps(noise.tolist())}')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
