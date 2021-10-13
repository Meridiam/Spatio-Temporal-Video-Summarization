import os
import time

import cv2
import imageio
from PIL import Image
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Get Object Centers")
    parser.add_argument(
        "--imgs_path", default="", help="world transform tagged points pathname"
    )
    parser.add_argument(
        "--save_path", default="FixImgs_outputs", help="where to save rgb images"
    )
    parser.add_argument(
        "--flip_y", default=False, type=bool, help="flip image along y-axis"
    )
    return parser

def main(args):
    save_folder = os.path.join(
        args.save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )
    os.makedirs(save_folder, exist_ok=True)

    imgs = [f for f in os.listdir(args.imgs_path)]
    ext = imgs[0].split(".")[1]

    i = 1
    for img in imgs:
        print(f'processing {i} / {len(imgs)}')
        if ext != "exr":
            data = cv2.imread(os.path.join(args.imgs_path, img), cv2.IMREAD_UNCHANGED)
            if args.flip_y:
                data = cv2.flip(data, 0)
            data = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(save_folder, f'{img.split("=")[1].split(".")[0]}.{ext}'), data)
        else:
            data = imageio.imread(os.path.join(args.imgs_path, img), "EXR-FI")
            if args.flip_y:
                data = cv2.flip(data, 0)
            data = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imageio.imsave(os.path.join(save_folder, f'{img.split("=")[1].split(".")[0]}.{ext}'), data)
        i += 1


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)