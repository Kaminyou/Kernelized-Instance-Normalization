import argparse
from pathlib import Path

from metrics.histogram import compare_images_histogram_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
parser.add_argument(
    "--image_A_path", type=str, required=True, help=("Path to the reference image")
)
parser.add_argument(
    "--image_B_path", type=str, required=True, help=("Path to the compared image")
)

if __name__ == "__main__":
    args = parser.parse_args()

    similarity = compare_images_histogram_pipeline(args.image_A_path, args.image_B_path)
    image_A_name = Path(args.image_A_path).stem
    image_B_name = Path(args.image_B_path).stem
    print(
        f"Exp::{args.exp_name}::{image_A_name} {image_B_name} || Histogram corr = {similarity:.4f}"
    )
