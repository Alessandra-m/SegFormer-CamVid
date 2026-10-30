import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping, _list_pairs, rgb_to_index


def main():
    mapping = build_label_mapping()
    num_labels = mapping["num_labels"]
    print(f"class_dict: {len(mapping['palette'])} строк, обучаемых классов {num_labels}, "
          f"игнор-класс={C.IGNORE_CLASS_NAME}")

    img_path, lbl_path = _list_pairs("train")[0]
    image = Image.open(img_path).convert("RGB")
    print(f"\nкартинка {img_path.name}: размер {image.size} (W,H)")

    if C.MASK_MODE == "rgb":
        mask_rgb = np.array(Image.open(lbl_path).convert("RGB"))
        idx = rgb_to_index(mask_rgb, mapping["palette"], mapping["row_to_train"])
        matched = (idx != C.IGNORE_INDEX).mean() * 100
        print(f"маска  {lbl_path.name}: размер {mask_rgb.shape[:2]} (H,W)")
        print(f"покрытие палитрой: {matched:.2f}% пикселей сопоставлены с классами")
    else:
        idx = np.array(Image.open(lbl_path)).astype(np.int64)
        print(f"маска  {lbl_path.name}: размер {idx.shape} (H,W), режим index")

    uniq = np.unique(idx)
    print("train-id в маске:", uniq.tolist())
    print("игнор-пикселей (255):", int((idx == C.IGNORE_INDEX).sum()))

    processor = AutoImageProcessor.from_pretrained(
        C.CHECKPOINT, do_reduce_labels=False,
        size={"height": C.IMAGE_SIZE, "width": C.IMAGE_SIZE},
    )
    ds = CamVidDataset("train", processor, mapping, augment=False)
    sample = ds[0]
    print("\nпосле image processor:")
    print("  pixel_values:", tuple(sample["pixel_values"].shape), sample["pixel_values"].dtype)
    print("  labels      :", tuple(sample["labels"].shape), sample["labels"].dtype)
    print("  уникальные labels:", torch.unique(sample["labels"]).tolist())

    model = AutoModelForSemanticSegmentation.from_pretrained(
        C.CHECKPOINT, num_labels=num_labels,
        id2label=mapping["id2label"], label2id=mapping["label2id"],
        ignore_mismatched_sizes=True,
    ).eval()
    with torch.no_grad():
        logits = model(pixel_values=sample["pixel_values"].unsqueeze(0)).logits
    print("\nлогиты модели:", tuple(logits.shape), "<- H/4 x W/4, НЕ полный размер")
    up = nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    print("после апсэмпла до исходного:", tuple(up.shape[-2:]), "= размер картинки", image.size[::-1])
    print("\nOK: обработка картинок/масок и восстановление размера работают как надо.")


if __name__ == "__main__":
    main()
