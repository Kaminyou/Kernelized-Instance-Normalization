import argparse
import os

import cv2
import numpy as np
from PIL import Image


def reduce_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size) * patch_size

def extend_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size + 1) * patch_size

if __name__ == "__main__":
    """
    USAGE
    1. prepare data belongs to domain X
        python3 crop.py -i ./data/example/HE_cropped.jpg -o ./data/example/trainX/ --thumbnail_output ./data/example/trainX/
    2. prepare data belongs to domain Y
        python3 crop.py -i ./data/example/ER_cropped.jpg -o ./data/example/trainY/ --thumbnail_output ./data/example/trainY/
    3. prepare data belongs to domain X required to be transferred to domain Y
        python3 crop.py -i ./data/example/HE_cropped.jpg -o ./data/example/testX/ --stride 512 --thumbnail_output ./data/example/testX/
    """
    parser = argparse.ArgumentParser(description="Crop a large image into patches.")
    parser.add_argument("-i","--input", help="Input image path", required=True)
    parser.add_argument("-o","--output", help="Output image path", default="data/initial/trainX/")
    parser.add_argument("--thumbnail", help="If crop a thumbnail or not", default=True)
    parser.add_argument("--thumbnail_output", help="Output image path", default="data/initial/")
    parser.add_argument("--patch_size", type=int, help="Patch size", default=512)
    parser.add_argument("--stride", type=int, help="Stride to crop patch", default=256)
    parser.add_argument("--mode", type=str, help="reduce or extend", default="reduce")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image = cv2.imread(args.input)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        raise ValueError

    if args.thumbnail:
        thumbnail = cv2.resize(image, (args.patch_size, args.patch_size), cv2.INTER_AREA)
        thumbnail_instance = Image.fromarray(thumbnail)
        thumbnail_instance.save(os.path.join(args.thumbnail_output, "thumbnail.png"))
    
    h, w, c = image.shape

    if args.mode == "reduce":
        resize_fn = reduce_size
        resize_code = cv2.INTER_AREA
    elif args.mode == "extend":
        resize_fn = extend_size
        resize_code = cv2.INTER_CUBIC
    else:
        raise NotImplementedError

    h_resize = resize_fn(h, args.patch_size)
    w_resize = resize_fn(w, args.patch_size)
    print(f"Original size: h={h} w={w}")
    print(f"Resize to: h={h_resize} w={w_resize}")

    image = cv2.resize(image, (w_resize, h_resize), resize_code)
    
    h_anchors = np.arange(0, h_resize, args.stride)
    w_anchors = np.arange(0, w_resize, args.stride)
    output_num = len(h_anchors) * len(w_anchors)
    max_idx_digits = max(len(str(len(h_anchors))), len(str(len(w_anchors))))
    max_anchor_digits = max(len(str(h_anchors[-1])), len(str(w_anchors[-1])))

    curr_idx = 1
    for y_idx, h_anchor in enumerate(h_anchors):
        for x_idx, w_anchor in enumerate(w_anchors):
            print(f"[{curr_idx} / {output_num}] Processing ...", end="\r")
            image_crop = image[h_anchor:h_anchor + args.patch_size, w_anchor:w_anchor + args.patch_size, :]
            image_crop_instance = Image.fromarray(image_crop)

            ## filename: {y-idx}_{x-idx}_{h-anchor}_{w-anchor}.png
            image_crop_instance.save(os.path.join(args.output, f"{str(y_idx).zfill(max_idx_digits)}_{str(x_idx).zfill(max_idx_digits)}_{str(h_anchor).zfill(max_anchor_digits)}_{str(w_anchor).zfill(max_anchor_digits)}.png"))
            curr_idx += 1


