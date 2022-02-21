import argparse
import os
from pathlib import Path

import cv2
from PIL import Image

from metrics.niqe import niqe
from metrics.piqe import piqe
from metrics.sobel import calculate_sobel_gradient_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="", help='Experiment name')
parser.add_argument('--path', type=str, required=True, help='Path to the image')
parser.add_argument('--save_grad', action='store_true', help='Whether to save the gardient image')

if __name__ == "__main__":
    args = parser.parse_args()
    sobel_gradient, sobel_gradient_avg = calculate_sobel_gradient_pipeline(args.path)

    im_bgr = cv2.imread(args.path)
    piqe_score, _, _, _ = piqe(im_bgr)
    niqe_score = niqe(im_bgr)

    img_path = Path(args.path)
    
    if args.save_grad:
        parent_path = img_path.parents[0]
        save_name = img_path.stem + "_grad.png"
        im = Image.fromarray(sobel_gradient)
        im.save(os.path.join(parent_path, save_name))
    
    print(f"Exp::{args.exp_name}::{img_path.stem} || Grad = {sobel_gradient_avg:.4f} PIQE = {piqe_score:.4f} NIQE = {niqe_score:.4f}")
