import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoImageProcessor, PrinterCallback, TrainerCallback, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

PRUNING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PRUNING_DIR.parent))

import config as C  # noqa: E402
from benchmark import load_test_set  # noqa: E402
from camvid_dataset import CamVidDataset, build_label_mapping, collate_fn  # noqa: E402
from dependency_groups import RESULTS, load_model  # noqa: E402
from perf_models import gpu_warmup  # noqa: E402
from prune_segformer import TARGETS, describe, measure, prune  # noqa: E402
from train import EpochLogger, SegTrainer, build_loss, build_optimizer, make_compute_metrics  # noqa: E402

WEIGHTS = C.PROJECT_DIR / "weights" / "segformer-b0-camvid"
OUTPUTS = PRUNING_DIR / "outputs"


def parse_spec(spec):
    if spec == "none":
        return []
    steps = []
    for item in spec.split(","):
        target, ratio = item.split(":")
        if target not in TARGETS:
            raise ValueError(f"unknown target {target}, expected one of {TARGETS}")
        steps.append((target, float(ratio)))
    return steps


def spec_name(spec):
    return "control" if spec == "none" else spec.replace(":", "").replace(",", "_")


def build_pruned(spec, importance="l1"):
    model = load_model()
    for target, ratio in parse_spec(spec):
        model = prune(model, target, ratio, importance)
    return model


def save_pruned(model, spec, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "state_dict.pt")
    (out_dir / "pruning.json").write_text(json.dumps({"spec": spec}))


def load_pruned(out_dir):
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "pruning.json").read_text())["spec"]
    model = build_pruned(spec)
    model.load_state_dict(torch.load(out_dir / "state_dict.pt", map_location="cpu"))
    return model.eval()


class BestState(TrainerCallback):
    def __init__(self):
        self.best = float("-inf")
        self.epoch = 0
        self.state = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics["eval_mean_iou"] > self.best:
            self.best = metrics["eval_mean_iou"]
            self.epoch = int(round(state.epoch))
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def finetune(model, processor, mapping, epochs, out_dir):
    train_ds = CamVidDataset("train", processor, mapping, train=True)
    val_ds = CamVidDataset("val", processor, mapping, train=False)
    total_steps = math.ceil(len(train_ds) / C.TRAIN_BATCH_SIZE) * epochs
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=C.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=C.EVAL_BATCH_SIZE,
        dataloader_num_workers=C.NUM_WORKERS,
        fp16=C.FP16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        remove_unused_columns=False,
        report_to="none",
        seed=C.SEED,
    )
    best = BestState()
    trainer = SegTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(mapping["num_labels"]),
        optimizers=build_optimizer(model, total_steps),
        loss_fn=build_loss(),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(EpochLogger())
    trainer.add_callback(best)
    trainer.train()
    model.load_state_dict(best.state)
    return model.eval(), best, trainer.state.log_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help='e.g. "decoder:0.5" "heads:0.5,residual4:0.25" or "none" for control')
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--importance", default="l1", choices=["l1", "random"])
    ap.add_argument("--compile", action="store_true", help="also measure fp16+compile latency")
    ap.add_argument("--tag", default="finetune")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU required"
    hf_logging.disable_progress_bar()
    torch.backends.cudnn.benchmark = True
    mapping = build_label_mapping()
    processor = AutoImageProcessor.from_pretrained(str(WEIGHTS))
    images, masks = load_test_set(mapping, "test")
    test_inputs = [processor(img, return_tensors="pt")["pixel_values"] for img in images]

    rows = []
    for spec in args.specs:
        set_seed(C.SEED)
        name = spec_name(spec)
        out_dir = OUTPUTS / name
        model = build_pruned(spec, args.importance)
        shape = "; ".join(describe(model, t) for t, _ in parse_spec(spec)) or "original"
        print(f"\n=== {name}: {shape} ===")

        gpu_warmup(1.0)
        before = measure(model, test_inputs, masks, mapping["num_labels"], False)
        print(f"before fine-tuning: test mIoU {before['miou']:.4f}")

        t0 = time.time()
        model, best, history = finetune(model, processor, mapping, args.epochs, out_dir)
        train_min = (time.time() - t0) / 60

        save_pruned(model, spec, out_dir)
        processor.save_pretrained(str(out_dir))
        gpu_warmup(1.0)
        after = measure(model, test_inputs, masks, mapping["num_labels"], args.compile)
        row = {"spec": spec, "name": name, "shape": shape, "epochs": args.epochs, "importance": args.importance,
               "best_epoch": best.epoch, "val_miou": best.best, "test_miou_before_ft": before["miou"],
               "train_min": train_min, **after}
        rows.append(row)
        with open(out_dir / "finetune_log.json", "w") as f:
            json.dump({**row, "log_history": history}, f, indent=2)
        print(f"{name}: val {best.best:.4f} (epoch {best.epoch}) | test {after['miou']:.4f} "
              f"(before ft {before['miou']:.4f}) | params {after['params_M']:.3f}M | {after['gmacs']:.2f} GMACs "
              f"| fp32 {after['latency_fp32_ms']:.2f} ms"
              + (f" | compile {after['latency_compile_ms']:.2f} ms" if "latency_compile_ms" in after else ""))

    RESULTS.mkdir(parents=True, exist_ok=True)
    head = ["Pruning", "Shape", "Params, M", "GMACs", "mIoU test before ft", "mIoU val (best)", "mIoU test",
            "Latency fp32, ms", "Latency fp16+compile, ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        compile_ms = r.get("latency_compile_ms")
        lines.append("| " + " | ".join([
            r["spec"], r["shape"], f"{r['params_M']:.3f}", f"{r['gmacs']:.2f}", f"{r['test_miou_before_ft']:.4f}",
            f"{r['val_miou']:.4f}", f"{r['miou']:.4f}", f"{r['latency_fp32_ms']:.2f}",
            "—" if compile_ms is None else f"{compile_ms:.2f}",
        ]) + " |")
    md = "\n".join(lines)
    (RESULTS / f"{args.tag}.md").write_text(md + "\n")
    (RESULTS / f"{args.tag}.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print("\n" + md)
    print(f"\nSaved: {RESULTS / (args.tag + '.md')}")


if __name__ == "__main__":
    main()
