import os
import time

import cv2
import imageio
import numpy as np
from PIL import Image
import argparse

from loguru import logger

def make_parser():
    parser = argparse.ArgumentParser("Get Object Centers")
    parser.add_argument(
        "--imgs_path", default="", help="world transform tagged points pathname"
    )
    parser.add_argument(
        "--save_path", default="RGB_Depth_outputs", help="where to save rgb and depth images"
    )
    return parser

def main(args):
    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    rgb_folder = os.path.join(save_folder, 'rgb')
    depth_folder = os.path.join(save_folder, 'depth')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    imgs = [f for f in os.listdir(args.imgs_path)]

    i = 1
    for img in imgs:
        data = cv2.imread(os.path.join(args.imgs_path, img), cv2.IMREAD_UNCHANGED)

        rgb_data = np.ndarray(shape=(data.shape[0], data.shape[1], 3), dtype=float)
        depth_data = np.ndarray(shape=(data.shape[0], data.shape[1]), dtype=int)
        depth_data = depth_data.astype("float32")

        for row_idx in range(len(data)):
            for col_idx in range(len(data[0])):
                rgb_data[row_idx][col_idx] = data[row_idx][col_idx][:-1]
                depth_data[row_idx][col_idx] = data[row_idx][col_idx][-1] / 100

        rgb_data = np.clip(np.power(rgb_data, 0.45), 0, 1)
        rgb_data = np.uint8(rgb_data*255)

        cv2.imwrite(os.path.join(rgb_folder, f'{img.split("frame")[1].split(".")[0]}.png'), rgb_data)
        imageio.imsave(os.path.join(depth_folder, f'{img.split("frame")[1].split(".")[0]}.exr'), depth_data)

        logger.info(f'Processed {i} / {len(imgs)} RGB-D images')
        i += 1


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)