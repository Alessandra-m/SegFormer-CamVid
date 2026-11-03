import json
import math
import time

import evaluate
import numpy as np
import torch
from segmentation_models_pytorch.losses import LovaszLoss
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup,
    set_seed,
)
from transformers.utils import logging as hf_logging

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping, collate_fn


def build_model(mapping):
    num_labels = mapping["num_labels"]
    label_kwargs = dict(id2label=mapping["id2label"], label2id=mapping["label2id"])
    if not C.HEAD_INIT_FROM:
        return AutoModelForSemanticSegmentation.from_pretrained(
            C.CHECKPOINT, num_labels=num_labels, ignore_mismatched_sizes=True, **label_kwargs
        )

    model = AutoModelForSemanticSegmentation.from_pretrained(C.CHECKPOINT)
    src_ids = {name: i for i, name in model.config.id2label.items()}
    rows = [src_ids[C.HEAD_INIT_FROM[mapping["id2label"][i]]] for i in range(num_labels)]

    old = model.decode_head.classifier
    new = nn.Conv2d(old.in_channels, num_labels, kernel_size=1)
    with torch.no_grad():
        new.weight.copy_(old.weight[rows])
        new.bias.copy_(old.bias[rows])
    model.decode_head.classifier = new
    model.config.num_labels = num_labels
    model.config.id2label = mapping["id2label"]
    model.config.label2id = mapping["label2id"]
    print(f"Classifier initialized from {C.CHECKPOINT} classes: {C.HEAD_INIT_FROM}")
    return model


def build_optimizer(model, total_steps):
    groups = {(is_head, decay): [] for is_head in (False, True) for decay in (False, True)}
    for name, param in model.named_parameters():
        groups[(name.startswith("decode_head"), param.ndim > 1)].append(param)
    param_groups = [
        {
            "params": params,
            "lr": C.LEARNING_RATE * (C.HEAD_LR_MULT if is_head else 1),
            "weight_decay": C.WEIGHT_DECAY if decay else 0.0,
        }
        for (is_head, decay), params in groups.items() if params
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=C.LEARNING_RATE)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, C.WARMUP_STEPS, total_steps, lr_end=0.0, power=1.0
    )
    return optimizer, scheduler


def build_loss():
    ce = nn.CrossEntropyLoss(ignore_index=C.IGNORE_INDEX)
    if C.LOSS == "ce":
        return ce
    if C.LOSS == "ce_lovasz":
        lovasz = LovaszLoss("multiclass", ignore_index=C.IGNORE_INDEX)
        return lambda logits, labels: ce(logits, labels) + lovasz(logits, labels)
    raise ValueError(f"Unknown LOSS: {C.LOSS}")


class EpochLogger(TrainerCallback):
    def __init__(self):
        self.train_loss = float("nan")
        self.best = float("-inf")
        self.start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        miou = metrics["eval_mean_iou"]
        self.best = max(self.best, miou)
        print(f"epoch {state.epoch:3.0f}/{args.num_train_epochs} | train loss {self.train_loss:.4f} | "
              f"val loss {metrics['eval_loss']:.4f} | val mIoU {miou:.4f} | best {self.best:.4f} | "
              f"{time.time() - self.start:.0f}s", flush=True)


class SegTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(pixel_values=inputs["pixel_values"])
        logits = nn.functional.interpolate(
            outputs.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(num_labels):
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        with torch.no_grad():
            preds = nn.functional.interpolate(
                torch.as_tensor(logits).float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
            ).argmax(dim=1).numpy().astype(np.int32)
        res = metric.compute(
            predictions=preds,
            references=labels.astype(np.int32),
            num_labels=num_labels,
            ignore_index=C.IGNORE_INDEX,
            reduce_labels=False,
        )
        return {
            "mean_iou": float(res["mean_iou"]),
            "mean_accuracy": float(res["mean_accuracy"]),
            "overall_accuracy": float(res["overall_accuracy"]),
        }

    return compute_metrics


def main():
    hf_logging.disable_progress_bar()
    set_seed(C.SEED)
    mapping = build_label_mapping()
    num_labels = mapping["num_labels"]
    print(f"Classes (excluding ignore): {num_labels} -> {mapping['id2label']}")

    processor = AutoImageProcessor.from_pretrained(
        C.CHECKPOINT,
        do_reduce_labels=False,
        size={"height": C.EVAL_SIZE[0], "width": C.EVAL_SIZE[1]},
    )
    model = build_model(mapping)

    train_ds = CamVidDataset("train", processor, mapping, train=True)
    val_ds = CamVidDataset("val", processor, mapping, train=False)
    print(f"train: {len(train_ds)} | val: {len(val_ds)} | eval input: {C.EVAL_SIZE} | crop: {C.CROP_SIZE}")

    total_steps = math.ceil(len(train_ds) / C.TRAIN_BATCH_SIZE) * C.NUM_EPOCHS
    args = TrainingArguments(
        output_dir=str(C.OUTPUT_DIR),
        num_train_epochs=C.NUM_EPOCHS,
        per_device_train_batch_size=C.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=C.EVAL_BATCH_SIZE,
        dataloader_num_workers=C.NUM_WORKERS,
        fp16=C.FP16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="epoch",
        disable_tqdm=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
        seed=C.SEED,
    )

    trainer = SegTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(num_labels),
        optimizers=build_optimizer(model, total_steps),
        loss_fn=build_loss(),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(EpochLogger())
    trainer.train()

    best = trainer.state.best_model_checkpoint
    best_model = AutoModelForSemanticSegmentation.from_pretrained(best)
    best_model.save_pretrained(str(C.OUTPUT_DIR))
    processor.save_pretrained(str(C.OUTPUT_DIR))
    with open(C.OUTPUT_DIR / "train_log.json", "w") as f:
        json.dump({"best_checkpoint": best, "best_val_miou": trainer.state.best_metric,
                   "log_history": trainer.state.log_history}, f, indent=2)

    print(f"\nDone. Best checkpoint {best} saved to {C.OUTPUT_DIR}")
    print(f"best val mIoU (during training): {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
