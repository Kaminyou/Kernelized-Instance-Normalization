import argparse
import os

import cv2
import numpy as np
from PIL import Image

from utils.util import read_yaml_config


def main():
    parser = argparse.ArgumentParser("Combined transferred images")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml", help="Path to the config file.")
    parser.add_argument("--patch_size", type=int, help="Patch size", default=512)
    parser.add_argument("--resize_h", type=int, help="Resize H", default=-1)
    parser.add_argument("--resize_w", type=int, help="Resize W", default=-1)
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    filenames = os.listdir(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", config['INFERENCE_SETTING']["NORMALIZATION"]))
    y_anchor_max = 0
    x_anchor_max = 0
    for filename in filenames:
        _, _, y_anchor, x_anchor, _ = filename.split("_", 4)
        y_anchor_max = max(y_anchor_max, int(y_anchor))
        x_anchor_max = max(x_anchor_max, int(x_anchor))

    matrix = np.zeros((y_anchor_max + args.patch_size, x_anchor_max + args.patch_size, 3), dtype=np.uint8)

    for filename in sorted(filenames):
        print(f"Combine {filename}  ", end="\r")
        _, _, y_anchor, x_anchor, _ = filename.split("_", 4)
        y_anchor = int(y_anchor)
        x_anchor = int(x_anchor)
        image = cv2.imread(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", "in", filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        matrix[y_anchor:y_anchor + 512, x_anchor:x_anchor + 512, :] = image

    if (args.resize_h != -1) and (args.resize_w != -1):
        matrix = cv2.resize(matrix, (args.resize_h, args.resize_w), cv2.INTER_CUBIC)

    matrix_image = Image.fromarray(matrix)
    matrix_image.save(os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test", f"combined_{config['INFERENCE_SETTING']['NORMALIZATION']}.png"))

if __name__ == "__main__":
    main()
