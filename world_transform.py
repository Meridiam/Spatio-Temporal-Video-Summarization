import argparse
import json
import os
import time
from loguru import logger

import cv2

from utils import parse_framewise_camera_pose, parse_intrinsics, image_to_world_space


def make_parser():
    parser = argparse.ArgumentParser("Get Object Centers")
    parser.add_argument(
        "--detections_path", default="", help="yolox detections folder pathname"
    )
    parser.add_argument(
        "--depth_path", default="", help="framewise depth map folder pathname"
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
    return parser


def main(args):
    if args.detections_path == "":
        print(f'Detections path name cannot be blank!')
        exit(1)

    if args.depth_path == "":
        print(f'Depth path name cannot be blank!')
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
    num_entries = sum(1 for _ in open(detections_pathname))
    with open(detections_pathname) as datafile:
        logger.info(f'Ingesting frame-level detections from {detections_pathname}')

        with open(os.path.join(save_folder, "tagged_points"), 'w+') as f:
            entry_idx = 0
            for line in datafile:
                frame_detections = json.loads(line)
                depthmap = cv2.imread(os.path.join(args.depth_path, f'{frame_detections["frame_idx"]}.exr'),
                                      cv2.IMREAD_UNCHANGED)

                for detection in frame_detections["data"]:
                    x_d, y_d = detection["center"]
                    depth = depthmap[int(y_d) // 8, int(x_d) // 8] # add 2 in 3rd dim when using original exrs

                    if depth > 0:
                        camera_pose = camera_poses[int(frame_detections["frame_idx"])]
                        assert int(frame_detections["frame_idx"]) == int(camera_pose["frame_idx"])

                        world_center3d = image_to_world_space(camera_pose, x_d, y_d, depth, fx_d, fy_d, cy_d, cx_d)

                        tagged_point = ({
                            "frame_idx": int(frame_detections["frame_idx"]),
                            "bbox": {
                                "p0": detection["bbox_p0"],
                                "p1": detection["bbox_p1"],
                                "center": detection["center"]
                            },
                            "class": detection["class"],
                            "score": detection["score"],
                            "worldCenter": world_center3d.tolist()
                        })

                        f.write(f'{json.dumps(tagged_point)}\n')

                entry_idx += 1
                logger.info(f'Processed entry {entry_idx} / {num_entries}')


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
