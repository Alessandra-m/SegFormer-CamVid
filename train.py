import json

import numpy as np
import torch
from torch import nn
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    set_seed,
)

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping, collate_fn


def main():
    set_seed(C.SEED)
    mapping = build_label_mapping()
    num_labels = mapping["num_labels"]
    print(f"Классов (без игнора): {num_labels} -> {mapping['id2label']}")

    processor = AutoImageProcessor.from_pretrained(
        C.CHECKPOINT,
        do_reduce_labels=False,        
        size={"height": C.IMAGE_SIZE, "width": C.IMAGE_SIZE},
    )

    model = AutoModelForSemanticSegmentation.from_pretrained(
        C.CHECKPOINT,
        num_labels=num_labels,
        id2label=mapping["id2label"],
        label2id=mapping["label2id"],
        ignore_mismatched_sizes=True,  
    )

    train_ds = CamVidDataset("train", processor, mapping, augment=True)
    val_ds = CamVidDataset("val", processor, mapping, augment=False)
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
            upsampled = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            ).argmax(dim=1)
            preds = upsampled.detach().cpu().numpy()
            res = metric.compute(
                predictions=preds,
                references=labels,
                num_labels=num_labels,
                ignore_index=C.IGNORE_INDEX,
                reduce_labels=False,
            )
            return {
                "mean_iou": float(res["mean_iou"]),
                "mean_accuracy": float(res["mean_accuracy"]),
                "overall_accuracy": float(res["overall_accuracy"]),
            }

    args = TrainingArguments(
        output_dir=str(C.OUTPUT_DIR),
        learning_rate=C.LEARNING_RATE,
        num_train_epochs=C.NUM_EPOCHS,
        per_device_train_batch_size=C.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=C.EVAL_BATCH_SIZE,
        dataloader_num_workers=C.NUM_WORKERS,
        eval_strategy="epoch",         
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        remove_unused_columns=False,    
        report_to="none",
        seed=C.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()


    final = trainer.evaluate()
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(C.OUTPUT_DIR))     
    processor.save_pretrained(str(C.OUTPUT_DIR))
    with open(C.OUTPUT_DIR / "val_metrics.json", "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\nГотово. Веса и processor сохранены в", C.OUTPUT_DIR)
    print("val mIoU:", round(final.get("eval_mean_iou", float("nan")), 4))


if __name__ == "__main__":
    main()
