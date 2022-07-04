import argparse
import os

import cv2

from utils.util import read_yaml_config


def main():
    """
    USAGE
        python3 transfer.py -c config_example.yaml
        or
        python3 transfer.py -c config_example.yaml --skip_cropping
    """
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument("--skip_cropping", action="store_true")
    args = parser.parse_args()

    config = read_yaml_config(args.config)
    H, W, _ = cv2.imread(config["INFERENCE_SETTING"]["TEST_X"]).shape
    if not args.skip_cropping:
        os.system(
            f"python3 crop.py -i {config['INFERENCE_SETTING']['TEST_X']} -o {config['INFERENCE_SETTING']['TEST_DIR_X']} --patch_size {config['CROPPING_SETTING']['PATCH_SIZE']} --stride {config['CROPPING_SETTING']['PATCH_SIZE']} --thumbnail_output {config['INFERENCE_SETTING']['TEST_DIR_X']}"
        )
        print("Finish cropping and start inference")
    os.system(f"python3 inference.py --config {args.config}")
    print("Finish inference and start combining images")
    os.system(
        f"python3 combine.py --config {args.config} --resize_h {H} --resize_w {W}"
    )


if __name__ == "__main__":
    main()
