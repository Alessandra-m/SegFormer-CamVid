import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor

import config as C
from camvid_dataset import CamVidDataset, _list_pairs, build_label_mapping
from train import build_model


def main():
    mapping = build_label_mapping()
    num_labels = mapping["num_labels"]
    print(f"class palette: {len(mapping['palette'])} rows, {num_labels} trainable classes, "
          f"ignore class = {C.IGNORE_CLASS_NAME}")

    processor = AutoImageProcessor.from_pretrained(
        C.CHECKPOINT,
        do_reduce_labels=False,
        size={"height": C.EVAL_SIZE[0], "width": C.EVAL_SIZE[1]},
    )
    ds = CamVidDataset("train", processor, mapping)

    img_path, lbl_path = _list_pairs("train")[0]
    image, idx = ds.original(0)
    print(f"\nimage {img_path.name}: {image.shape[:2]} (H, W)")
    print(f"mask  {lbl_path.name}: {idx.shape} (H, W), mode {C.MASK_MODE}")
    if C.MASK_MODE == "rgb":
        colors = np.unique(np.array(Image.open(lbl_path).convert("RGB")).reshape(-1, 3), axis=0)
        known = {tuple(c) for c in mapping["palette"].tolist()}
        unknown = [tuple(c) for c in colors.tolist() if tuple(c) not in known]
        print(f"mask colors: {len(colors)}, not in palette: {len(unknown)}")
    print("train ids in mask:", np.unique(idx).tolist())
    print(f"ignored pixels ({C.IGNORE_INDEX}): {(idx == C.IGNORE_INDEX).mean() * 100:.2f}%")

    train_sample = CamVidDataset("train", processor, mapping, train=True)[0]
    print(f"\ntrain sample (random scale {C.SCALE_RANGE} + crop {C.CROP_SIZE}):")
    print("  pixel_values:", tuple(train_sample["pixel_values"].shape))
    print("  labels      :", tuple(train_sample["labels"].shape), "unique:", torch.unique(train_sample["labels"]).tolist())

    sample = ds[0]
    print("\neval sample (image resized, labels kept at native size):")
    print("  pixel_values:", tuple(sample["pixel_values"].shape), sample["pixel_values"].dtype)
    print("  labels      :", tuple(sample["labels"].shape), sample["labels"].dtype)
    print("  unique labels:", torch.unique(sample["labels"]).tolist())

    model = build_model(mapping).eval()
    with torch.no_grad():
        logits = model(pixel_values=sample["pixel_values"].unsqueeze(0)).logits
    print("\nmodel logits:", tuple(logits.shape), "(H/4 x W/4, not full size)")

    up = nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    print("after upsampling:", tuple(up.shape[-2:]), "= image size", image.shape[:2])
    assert up.shape[-2:] == idx.shape
    print("\nOK: image/mask processing and prediction resizing work as expected.")


if __name__ == "__main__":
    main()
