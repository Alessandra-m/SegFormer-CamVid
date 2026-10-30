import argparse
import json
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping, colorize


def predict_original(model, processor, image, device, flip=False):
    pixel_values = processor(image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        if flip:
            logits = logits + model(pixel_values=pixel_values.flip(-1)).logits.flip(-1)
        logits = nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
    return logits.argmax(dim=1)[0].cpu().numpy()


def per_image_iou(pred, gt, num_labels):
    valid = gt != C.IGNORE_INDEX
    ious = []
    for c in range(num_labels):
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(np.logical_and(p, g).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


def evaluate_split(split, model, processor, mapping, device, flip=False):
    num_labels = mapping["num_labels"]
    ds = CamVidDataset(split, processor, mapping)
    metric = evaluate.load("mean_iou")

    per_img = []
    for i in range(len(ds)):
        image, gt = ds.original(i)
        pred = predict_original(model, processor, image, device, flip)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)
        metric.add(prediction=pred.astype(np.int32), reference=gt.astype(np.int32))
        per_img.append((per_image_iou(pred, gt, num_labels), i))
        if (i + 1) % 50 == 0:
            print(f"  {split}: {i + 1}/{len(ds)} processed")

    res = metric.compute(num_labels=num_labels, ignore_index=C.IGNORE_INDEX, reduce_labels=False)
    summary = {
        "mean_iou": float(res["mean_iou"]),
        "mean_accuracy": float(res["mean_accuracy"]),
        "overall_accuracy": float(res["overall_accuracy"]),
        "per_class_iou": {mapping["id2label"][c]: float(v) for c, v in enumerate(res["per_category_iou"])},
        "n_images": len(ds),
        "resolution": list(gt.shape),
    }
    return summary, ds, per_img


def error_map(pred, gt):
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    out[pred != gt] = (220, 30, 30)
    out[gt == C.IGNORE_INDEX] = (110, 110, 110)
    return out


def save_examples(ds, model, processor, mapping, device, flip, items, tag, out_dir):
    for rank, (iou, idx) in enumerate(items, 1):
        img_path, _ = ds.pairs[idx]
        image, gt = ds.original(idx)
        pred = predict_original(model, processor, image, device, flip)
        valid = gt != C.IGNORE_INDEX
        err = float((pred[valid] != gt[valid]).mean() * 100)

        panels = [
            (image, "Image"),
            (colorize(gt, mapping["train_palette"]), "Ground truth"),
            (colorize(pred, mapping["train_palette"]), f"Prediction (image mIoU = {iou:.3f})"),
            (error_map(pred, gt), f"Errors: {err:.1f}% of pixels"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{tag} #{rank}: {img_path.name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_{rank}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=C.OUTPUT_DIR)
    parser.add_argument("--flip", action="store_true", help="horizontal flip test-time augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = build_label_mapping()
    processor = AutoImageProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForSemanticSegmentation.from_pretrained(str(args.model_dir)).to(device).eval()
    print(f"model: {args.model_dir} | input: {processor.size['height']}x{processor.size['width']} | flip TTA: {args.flip}")

    results = {}
    for split in ("val", "test"):
        results[split], test_ds, per_img = evaluate_split(split, model, processor, mapping, device, args.flip)

    suffix = "_flip" if args.flip else ""
    with open(args.model_dir / f"metrics{suffix}.json", "w") as f:
        json.dump({"model_dir": str(args.model_dir), "flip": args.flip, **results}, f, indent=2)

    for split, s in results.items():
        h, w = s["resolution"]
        print(f"\n=== {split} ({s['n_images']} images, {h}x{w}) ===")
        print(f"mIoU            : {s['mean_iou']:.4f}")
        print(f"mean accuracy   : {s['mean_accuracy']:.4f}")
        print(f"overall accuracy: {s['overall_accuracy']:.4f}")
    print("\nper-class IoU        val     test")
    for name in results["test"]["per_class_iou"]:
        print(f"    {name:<16} {results['val']['per_class_iou'][name]:.4f}  "
              f"{results['test']['per_class_iou'][name]:.4f}")

    vis_dir = args.model_dir / f"predictions{suffix}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    per_img.sort(key=lambda t: np.inf if np.isnan(t[0]) else t[0])
    save_examples(test_ds, model, processor, mapping, device, args.flip, per_img[-C.N_BEST:][::-1], "best", vis_dir)
    save_examples(test_ds, model, processor, mapping, device, args.flip, per_img[:C.N_WORST], "worst", vis_dir)
    print(f"\nExamples saved to {vis_dir} (best_*.png, worst_*.png)")


if __name__ == "__main__":
    main()
