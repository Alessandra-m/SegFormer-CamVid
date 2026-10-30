import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import evaluate
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping, colorize


def predict_index_map(model, pixel_values, out_hw, device):
    """pixel_values (1,3,H,W) -> карта индексов (out_hw) после апсэмпла логитов."""
    with torch.no_grad():
        logits = model(pixel_values=pixel_values.to(device)).logits   # (1, K, H/4, W/4)
        logits = nn.functional.interpolate(
            logits, size=out_hw, mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1)[0].cpu().numpy()


def predict_original(model, processor, image, device):
    """Картинка (H,W,3) -> карта индексов того же размера (H,W).

    processor сжимает картинку до IMAGE_SIZE x IMAGE_SIZE, модель выдаёт логиты H/4 x W/4
    от этого размера, логиты апсэмплятся обратно до исходного разрешения кадра.
    """
    enc = processor(image, return_tensors="pt")
    return predict_index_map(model, enc["pixel_values"], image.shape[:2], device)


def per_image_iou(pred, gt, num_labels, ignore_index):
    """Средний по присутствующим классам IoU для одного изображения."""
    ious = []
    for c in range(num_labels):
        valid = gt != ignore_index
        p = (pred == c) & valid
        g = (gt == c) & valid
        if g.sum() == 0 and p.sum() == 0:
            continue
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def evaluate_split(split, model, processor, mapping, device):
    """mIoU / accuracy / per-class IoU на сплите в исходном разрешении масок (360x480)."""
    num_labels = mapping["num_labels"]
    ds = CamVidDataset(split, processor, mapping, augment=False)
    metric = evaluate.load("mean_iou")

    per_img = []            # (iou, index)
    for i in range(len(ds)):
        image, gt = ds.original(i)
        pred = predict_original(model, processor, image, device)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)
        metric.add(prediction=pred, reference=gt)
        per_img.append((per_image_iou(pred, gt, num_labels, C.IGNORE_INDEX), i))
        if (i + 1) % 50 == 0:
            print(f"  {split}: обработано {i + 1}/{len(ds)}")

    res = metric.compute(num_labels=num_labels, ignore_index=C.IGNORE_INDEX, reduce_labels=False)
    per_class = {mapping["id2label"][c]: float(v) for c, v in enumerate(res["per_category_iou"])}
    summary = {
        "mean_iou": float(res["mean_iou"]),
        "mean_accuracy": float(res["mean_accuracy"]),
        "overall_accuracy": float(res["overall_accuracy"]),
        "per_class_iou": per_class,
        "n_images": len(ds),
        "resolution": list(gt.shape),
    }
    return summary, ds, per_img


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else C.OUTPUT_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = build_label_mapping()

    processor = AutoImageProcessor.from_pretrained(str(C.OUTPUT_DIR))
    model = AutoModelForSemanticSegmentation.from_pretrained(str(model_dir)).to(device).eval()
    print("модель:", model_dir)

    results = {}
    for split in ("val", "test"):
        results[split], ds, per_img = evaluate_split(split, model, processor, mapping, device)
    test_ds = ds

    with open(C.OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump({"model_dir": str(model_dir), **results}, f, indent=2, ensure_ascii=False)

    for split, s in results.items():
        print(f"\n=== {split} ({s['n_images']} кадров, {s['resolution'][0]}x{s['resolution'][1]}) ===")
        print("mIoU            :", round(s["mean_iou"], 4))
        print("mean accuracy   :", round(s["mean_accuracy"], 4))
        print("overall accuracy:", round(s["overall_accuracy"], 4))
    print("\nper-class IoU   :    val     test")
    for name in results["test"]["per_class_iou"]:
        print(f"    {name:<16} {results['val']['per_class_iou'][name]:.4f}  {results['test']['per_class_iou'][name]:.4f}")

    # лучшие и худшие примеры на test
    C.VIS_DIR.mkdir(parents=True, exist_ok=True)
    per_img.sort(key=lambda t: (np.inf if np.isnan(t[0]) else t[0]))
    worst = per_img[:C.N_WORST]
    best = per_img[-C.N_BEST:][::-1]
    _save_examples(test_ds, model, processor, mapping, device, best, "best")
    _save_examples(test_ds, model, processor, mapping, device, worst, "worst")
    print(f"\nПримеры сохранены в {C.VIS_DIR} (best_*.png, worst_*.png)")


def _error_map(pred, gt):
    """Красный — неверные пиксели, серый — игнорируемые (unlabelled), чёрный — верные."""
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    out[pred != gt] = (220, 30, 30)
    out[gt == C.IGNORE_INDEX] = (110, 110, 110)
    return out


def _save_examples(test_ds, model, processor, mapping, device, items, tag):
    for rank, (iou, idx) in enumerate(items, 1):
        img_path, _ = test_ds.pairs[idx]
        image, gt = test_ds.original(idx)
        pred = predict_original(model, processor, image, device)
        valid = gt != C.IGNORE_INDEX
        err = float((pred[valid] != gt[valid]).mean() * 100)

        fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
        ax[0].imshow(image);                                    ax[0].set_title("Изображение")
        ax[1].imshow(colorize(gt, mapping["train_palette"]));   ax[1].set_title("Эталон (GT)")
        ax[2].imshow(colorize(pred, mapping["train_palette"])); ax[2].set_title(f"Предсказание (mIoU кадра={iou:.3f})")
        ax[3].imshow(_error_map(pred, gt));                     ax[3].set_title(f"Ошибки: {err:.1f}% пикселей")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{tag} #{rank}: {img_path.name}")
        fig.tight_layout()
        fig.savefig(C.VIS_DIR / f"{tag}_{rank}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
