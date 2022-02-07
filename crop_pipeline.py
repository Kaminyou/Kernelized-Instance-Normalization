import argparse
import os

from utils.util import read_yaml_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument("-c", "--config", type=str, default="./data/example/config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = read_yaml_config(args.config)['CROPPING_SETTING']
    os.system(f"python3 crop.py -i {config['TRAIN_X']} -o {config['TRAIN_DIR_X']} --patch_size {config['PATCH_SIZE']} --stride {config['STRIDE']}")
    os.system(f"python3 crop.py -i {config['TRAIN_Y']} -o {config['TRAIN_DIR_Y']} --patch_size {config['PATCH_SIZE']} --stride {config['STRIDE']}")
    os.system(f"python3 crop.py -i {config['TEST_X']} -o {config['TEST_DIR_X']} --thumbnail --thumbnail_output {config['TEST_DIR_X']} --patch_size {config['PATCH_SIZE']} --stride {config['PATCH_SIZE']}")
