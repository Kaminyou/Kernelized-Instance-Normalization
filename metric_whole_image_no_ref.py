import argparse
import os
from pathlib import Path

from PIL import Image

from metrics.sobel import calculate_sobel_gradient_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Path to the image')
parser.add_argument('--save_grad', action='store_true', help='Whether to save the gardient image')

if __name__ == "__main__":
    args = parser.parse_args()
    sobel_gradient, sobel_gradient_avg = calculate_sobel_gradient_pipeline(args.path)

    img_path = Path(args.path)
    
    if args.save_grad:
        parent_path = img_path.parents[0]
        save_name = img_path.stem + "_grad.png"
        im = Image.fromarray(sobel_gradient)
        im.save(os.path.join(parent_path, save_name))
    
    print(f"{img_path.stem} || Grad = {sobel_gradient_avg:.4f}")
