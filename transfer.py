import argparse
import os

import cv2

from utils.util import read_yaml_config


def main():
    """
    USAGE
        python3 transfer.py -c config_example.yaml -i ./data/example/HE_cropped.jpg -o ./data/example/testX/ 
        python3 transfer.py -c config_example.yaml -i ./data/example/HE_cropped.jpg --skip_cropping
    """
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml", help="Path to the config file.")
    parser.add_argument("-i", "--image", type=str, required=True)
    parser.add_argument("-o", "--cropped_output", type=str, default=None)
    parser.add_argument("--skip_cropping", action="store_true")
    args = parser.parse_args()

    config = read_yaml_config(args.config)
    H, W, _ = cv2.imread(args.image).shape
    if not args.skip_cropping:
        assert args.cropped_output != None
        os.system(f"python3 crop.py -i {args.image} -o {args.cropped_output} --stride 512 --thumbnail_output {config['INFERENCE_SETTING']['TEST_DIR_X']}")
        print("Finish cropping and start inference")
    os.system(f"python3 inference.py --config {args.config}")
    print("Finish inference and start combining images")
    os.system(f"python3 combine.py --config {args.config} --resize_h {H} --resize_w {W}")

if __name__ == "__main__":
    main()
