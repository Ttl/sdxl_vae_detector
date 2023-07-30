#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import exif_transpose
from vae_detector import VAEDetector
import sys
import glob
import os
import numpy as np

detector_file = "sdxl_vae_detector_10.pt"
device = torch.device("cuda")

def to_tensor(img):
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img).to(dtype=torch.float)
    img = img.to(torch.float) / 127.5 - 1
    img = torch.unsqueeze(img, 0)
    return img

if __name__ == "__main__":
    detector = VAEDetector()
    detector.load_state_dict(torch.load("sdxl_vae_detector_10.pt", map_location=device))
    detector.eval()
    detector.to(device)
    imgs = glob.glob(sys.argv[1])
    pos, neg = 0, 0
    th = 0.5
    scores = []
    for img_path in imgs:
        img_name = os.path.split(img_path)[1]
        img = Image.open(img_path).convert("RGB")
        img = exif_transpose(img)
        img = to_tensor(np.asarray(img))
        with torch.inference_mode():
            pred = detector(img.to(device))
            pred = pred.cpu().numpy().item()
            print(img_name, pred)
            if pred >= th:
                pos += 1
            else:
                neg += 1
            scores.append(pred)
    print(pos, neg)
