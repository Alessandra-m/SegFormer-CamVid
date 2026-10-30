import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import config as C
from camvid_dataset import build_label_mapping, colorize


def main():
    if len(sys.argv) < 2:
        print("usage: python3 predict.py <image.png> [out.png]")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else img_path.with_name(img_path.stem + "_pred.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = build_label_mapping()
    processor = AutoImageProcessor.from_pretrained(str(C.OUTPUT_DIR))
    model = AutoModelForSemanticSegmentation.from_pretrained(str(C.OUTPUT_DIR)).to(device).eval()

    image = Image.open(img_path).convert("RGB")
    enc = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(pixel_values=enc["pixel_values"].to(device)).logits   # (1, K, H/4, W/4)
    print("размер логитов (H/4 x W/4):", tuple(logits.shape[-2:]))

    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False   # (W,H)->(H,W)
    )
    print("после апсэмпла до исходного:", tuple(upsampled.shape[-2:]), "= размер картинки", image.size[::-1])
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()

    color = colorize(pred, mapping["train_palette"])
    overlay = (np.array(image) * 0.5 + color * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
