import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


def parse_framewise_camera_pose(framewise_pathname):
    camera_poses = {}
    num_entries = 0
    with open(framewise_pathname) as f:
        logger.info(f'Parsing framewise camera poses from {framewise_pathname}')
        for line in f:
            if num_entries > 0:
                entries = line.split(" ")

                camera_poses[int(entries[1])] = {
                    "frame_idx": int(entries[1]),
                    "x": float(entries[3]),
                    "y": float(entries[4]),
                    "z": float(entries[5]),
                    "roll": float(entries[6]),
                    "pitch": float(entries[7]),
                    "yaw": float(entries[8])
                }

            num_entries += 1

    logger.info(f'Parsed {num_entries} camera poses')
    return camera_poses


def parse_intrinsics(intrinsics_pathname):
    with open(intrinsics_pathname) as f:
        logger.info(f'Parsing camera intrinsics from {intrinsics_pathname}')

        intrinsics = f.readlines()[1].split(" ")

        logger.info(
            f'Camera intrinsics - focalLengthX: {intrinsics[0]}, focalLengthY: {intrinsics[1]}, '
            f'principalPointX: {intrinsics[2]}, principalPointY: {intrinsics[3]}'
        )

        return float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])


def apply_camera_transform(camera_pose, point):
    rot = Rotation.from_euler('xyz', [camera_pose["roll"], camera_pose["pitch"], camera_pose["yaw"]], degrees=True)

    return rot.apply(point)


def image_to_world_space(camera_pose, x_d, y_d, depth, fx_d, fy_d, cy_d, cx_d):
    local_center3d = np.array([
        (x_d - cy_d) * depth / fy_d,
        (y_d - cx_d) * depth / fx_d,
        depth
    ])
    world_center3d = apply_camera_transform(camera_pose, local_center3d) \
                     + np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])
    return world_center3d


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
