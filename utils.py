from loguru import logger


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
