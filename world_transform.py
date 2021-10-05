import argparse
import json
import os
import time
from loguru import logger

from scipy.spatial.transform import Rotation
import numpy as np

import cv2


def make_parser():
    parser = argparse.ArgumentParser("Get Object Centers")
    parser.add_argument(
        "--detections_path", default="", help="yolox detections folder pathname"
    )
    parser.add_argument(
        "--depth_path", default="", help="framewise depth map folder pathname"
    )
    parser.add_argument(
        "--confidence_path", default="", help="framewise confidence map folder pathname"
    )
    parser.add_argument(
        "--framewise_path", default="", help="framewise camera pose file pathname"
    )
    parser.add_argument(
        "--intrinsics_path", default="", help="camera intrinsics file pathname"
    )
    parser.add_argument(
        "--save_path", default="WorldTransform_outputs", help="pathname for results folder"
    )
    parser.add_argument(
        "--conf", type=float, default=0.0, help="minimum acceptable confidence level for depth"
    )
    return parser


def parse_intrinsics(intrinsics_pathname):
    with open(intrinsics_pathname) as f:
        logger.info(f'Parsing camera intrinsics from {intrinsics_pathname}')

        intrinsics = f.readlines()[1].split(" ")

        logger.info(
            f'Camera intrinsics - focalLengthX: {intrinsics[0]}, focalLengthY: {intrinsics[1]}, '
            f'principalPointX: {intrinsics[2]}, principalPointY: {intrinsics[3]}'
        )

        return float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])


def parse_framewise_camera_pose(framewise_pathname):
    camera_poses = []
    num_entries = 0
    with open(framewise_pathname) as f:
        logger.info(f'Parsing framewise camera poses from {framewise_pathname}')
        for line in f:
            if num_entries > 0:
                entries = line.split(" ")

                camera_poses.append({
                    "frame_idx": int(entries[1]),
                    "x": float(entries[3]),
                    "y": float(entries[4]),
                    "z": float(entries[5]),
                    "roll": float(entries[6]),
                    "pitch": float(entries[7]),
                    "yaw": float(entries[8])
                })

            num_entries += 1

    logger.info(f'Parsed {num_entries} camera poses')
    return camera_poses


def apply_camera_transform(camera_pose, point):
    rot = Rotation.from_euler('xyz', [camera_pose["roll"], camera_pose["pitch"], camera_pose["yaw"]], degrees=True)

    return rot.apply(point)


def main(args):
    if args.detections_path == "":
        print(f'Detections path name cannot be blank!')
        exit(1)

    if args.depth_path == "":
        print(f'Depth path name cannot be blank!')
        exit(1)

    if args.confidence_path == "":
        print(f'Confidence path name cannot be blank!')
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

    logger.info(f'Save location configured as {save_folder}')

    fx_d, fy_d, cx_d, cy_d = parse_intrinsics(args.intrinsics_path)

    camera_poses = parse_framewise_camera_pose(args.framewise_path)

    detections_pathname = os.path.join(args.detections_path, 'detections')
    num_entries = sum(1 for line in open(detections_pathname))
    with open(detections_pathname) as datafile:
        logger.info(f'Ingesting frame-level detections from {detections_pathname}')

        with open(os.path.join(save_folder, "tagged_points"), 'w+') as f:
            entry_idx = 0
            for line in datafile:
                t0 = time.time()

                frame_detections = json.loads(line)
                depthmap = cv2.imread(os.path.join(args.depth_path, f'frame={frame_detections["frame_idx"]}.exr'),
                                      cv2.IMREAD_UNCHANGED)
                confidencemap = cv2.imread(os.path.join(args.confidence_path,
                                    f'frame={frame_detections["frame_idx"]}.exr'),
                                    cv2.IMREAD_UNCHANGED)

                for detection in frame_detections["data"]:
                    x_d, y_d = detection["center"]
                    depth = depthmap[int(x_d) // 8, int(y_d) // 8, 2] * 100

                    if confidencemap[int(x_d) // 8, int(y_d) // 8, 2] > args.conf:
                        local_center3d = np.array([
                            (x_d - cx_d) * depth / fx_d,
                            (y_d - cy_d) * depth / fy_d,
                            depth
                        ])

                        camera_pose = camera_poses[int(frame_detections["frame_idx"])]

                        assert int(frame_detections["frame_idx"]) == int(camera_pose["frame_idx"])

                        world_center3d = apply_camera_transform(camera_pose, local_center3d) \
                            + np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])

                        tagged_point = ({
                            "frame_idx": int(frame_detections["frame_idx"]),
                            "bbox": {
                                "p0": detection["bbox_p0"],
                                "p1": detection["bbox_p1"],
                                "center": detection["center"]
                            },
                            "class": detection["class"],
                            "worldCenter": world_center3d.tolist()
                        })

                        f.write(f'{json.dumps(tagged_point)}\n')

                entry_idx += 1
                logger.info(f'Processed entry {entry_idx} / {num_entries}')

if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
