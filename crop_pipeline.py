# Author: Kaminyou (https://github.com/Kaminyou)
import argparse
import os

from utils.util import read_yaml_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = read_yaml_config(args.config)["CROPPING_SETTING"]
    # Prepare the training set
    for train_x, train_y in zip(config["TRAIN_X"], config["TRAIN_Y"]):
        os.system(
            f"python3 crop.py -i {train_x} -o {config['TRAIN_DIR_X']} "
            f"--patch_size {config['PATCH_SIZE']} --stride {config['STRIDE']}"
        )
        os.system(
            f"python3 crop.py -i {train_y} -o {config['TRAIN_DIR_Y']} "
            f"--patch_size {config['PATCH_SIZE']} --stride {config['STRIDE']}"
        )

    # Prepare the testing set
    for test_x in config["TEST_X"]:
        base = os.path.basename(test_x)
        filename = os.path.splitext(base)[0]
        output_path = os.path.join(config['TEST_DIR_X'], filename)
        os.system(
            f"python3 crop.py -i {test_x} -o {output_path} "
            f"--thumbnail --thumbnail_output {output_path} "
            f"--patch_size {config['PATCH_SIZE']} "
            f"--stride {config['PATCH_SIZE']}"
        )
