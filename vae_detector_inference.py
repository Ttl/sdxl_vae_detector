#!/usr/bin/env python
import torch
from PIL import Image, UnidentifiedImageError
from PIL.ImageOps import exif_transpose
from vae_detector import VAEDetector
import sys
import glob
import os
import numpy as np
import argparse

detector_file = "sdxl_vae_detector.pt"
device = torch.device("cuda")

def to_tensor(img):
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img).to(dtype=torch.float)
    img = img.to(torch.float) / 127.5 - 1
    img = torch.unsqueeze(img, 0)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE artifact detector inference.")
    parser.add_argument(
        "path",
        type=str,
        help="Images to detect. Supports glob pattern.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=detector_file,
        required=False,
        help="Neural net weight file to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Summary of number of positive/negative detections.",
    )
    parser.add_argument(
        "--print",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print detection for each input image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold",
    )
    args = parser.parse_args()

    detector = VAEDetector()
    detector.load_state_dict(torch.load(detector_file, map_location=args.device))
    detector.eval()
    detector.to(args.device)
    imgs = glob.glob(args.path)
    pos, neg = 0, 0
    for img_path in imgs:
        img_name = os.path.split(img_path)[1]
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f'Failed to open image {img_path}', file=sys.stderr)
            continue
        img = exif_transpose(img)
        img = to_tensor(np.asarray(img))
        with torch.inference_mode():
            pred = detector(img.to(device))
            pred = pred.cpu().numpy().item()
            if args.print:
                print(img_name, '{:.3f}'.format(pred))
            if pred >= args.threshold:
                pos += 1
            else:
                neg += 1
    if args.summary:
        print('Positive', pos, 'Negative', neg)
