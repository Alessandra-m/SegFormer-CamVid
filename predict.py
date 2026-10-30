import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import config as C
from camvid_dataset import build_label_mapping, colorize
from evaluate_model import predict_original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--model", type=Path, default=C.OUTPUT_DIR)
    parser.add_argument("--flip", action="store_true", help="horizontal flip test-time augmentation")
    args = parser.parse_args()
    out_path = args.out or args.image.with_name(args.image.stem + "_pred.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = build_label_mapping()
    processor = AutoImageProcessor.from_pretrained(str(args.model))
    model = AutoModelForSemanticSegmentation.from_pretrained(str(args.model)).to(device).eval()

    image = np.array(Image.open(args.image).convert("RGB"))
    pred = predict_original(model, processor, image, device, args.flip)
    print(f"image size: {image.shape[:2]}, prediction size: {pred.shape}")

    overlay = (image * 0.5 + colorize(pred, mapping["train_palette"]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
