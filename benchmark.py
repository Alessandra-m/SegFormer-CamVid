import argparse
import json
import time

import numpy as np
import torch

import config as C
from camvid_dataset import CamVidDataset, build_label_mapping
from perf_models import (VARIANT_INFO, VARIANTS, Variant, add_model_args, count_gmacs, count_params,
                         env_info, gpu_warmup, load, measure_latency, measure_memory, measure_throughput,
                         parse_size, resolve_model_args)

OUT_DIR = C.PROJECT_DIR / "results" / "perf"


def load_test_set(mapping, split):
    ds = CamVidDataset(split, processor=None, mapping=mapping)
    images, masks = [], []
    for i in range(len(ds)):
        img, gt = ds.original(i)
        images.append(img)
        masks.append(gt.astype(np.uint8))
    return images, masks


@torch.inference_mode()
def eval_miou(fn, inputs, masks, num_labels, device="cuda"):
    n = num_labels
    cm = torch.zeros(n * n, dtype=torch.long, device=device)
    for x, gt in zip(inputs, masks):
        pred = fn(x.to(device, non_blocking=True)).argmax(1)[0]
        gt = torch.from_numpy(gt).to(device).long()
        valid = gt != C.IGNORE_INDEX
        cm += torch.bincount(gt[valid] * n + pred[valid], minlength=n * n)
    cm = cm.view(n, n).double().cpu()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    return float(np.nanmean(iou.numpy())), [float(v) for v in iou], float(tp.sum() / cm.sum())


def fmt(v, f):
    return "—" if v is None else f.format(v)


def to_markdown(rows, batch):
    head = ["Model", "Input", "Variant", "mIoU test", "Δ mIoU", "Latency b=1, ms (median / p90)",
            "Speedup", f"FPS, b={batch}", "Memory b=1, MB (weights + activations)", "Params, M", "GMACs"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['input']} | {r['variant']} | error: {r['error']} |"
                         + " |" * (len(head) - 4))
            continue
        lat = r["latency_b1"]
        lines.append("| " + " | ".join([
            r["model"], r["input"], r["variant"],
            fmt(r.get("miou"), "{:.4f}"), fmt(r.get("delta_miou"), "{:+.4f}"),
            f"{lat['median_ms']:.2f} / {lat['p90_ms']:.2f}",
            fmt(r.get("speedup"), "×{:.2f}"),
            fmt(r.get("throughput_fps"), "{:.1f}"),
            "—" if r["memory_b1_MB"] is None else
            f"{r['weights_MB'] + r['memory_b1_MB']:.0f} ({r['weights_MB']:.0f} + {r['memory_b1_MB']:.0f})",
            f"{r['params_M']:.2f}", fmt(r.get("gmacs"), "{:.1f}"),
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--models", nargs="+", default=["segformer", "deeplab"], choices=["segformer", "deeplab"])
    ap.add_argument("--sizes", nargs="+", default=["360x480"], help="input sizes HxW")
    ap.add_argument("--variants", nargs="+", default=["fp32", "tf32", "fp16", "bf16", "half"],
                    choices=VARIANTS)
    ap.add_argument("--batch", type=int, default=8, help="batch size for throughput")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--miou", action="store_true", help="compute mIoU of every variant on --split")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tag", default="benchmark", help="result file name in results/perf/")
    args = resolve_model_args(ap.parse_args())

    assert torch.cuda.is_available(), "GPU required"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    sizes = [parse_size(s) for s in args.sizes]
    mapping = build_label_mapping()
    num_labels = mapping["num_labels"]

    if args.miou:
        t0 = time.time()
        images, masks = load_test_set(mapping, args.split)
        print(f"{args.split}: {len(images)} images loaded in {time.time() - t0:.0f} s")

    env = env_info()
    print("environment:", json.dumps(env))
    gpu_warmup()

    rows = []
    for model_name in args.models:
        loaded = load(model_name, args)
        net = loaded.net.to(device).eval()
        params_m = count_params(net) / 1e6
        for size in sizes:
            in_str = f"{size[0]}x{size[1]}"
            inputs = None
            if args.miou:
                t0 = time.time()
                inputs = [loaded.preprocess(img, size) for img in images]
                print(f"\n[{loaded.name} {in_str}] preprocessed {len(inputs)} images in {time.time() - t0:.0f} s")
            try:
                gmacs = count_gmacs(net, size)
            except Exception as e:
                print("  GMACs not counted:", e)
                gmacs = None

            base = None
            for vname in args.variants:
                row = {"model": loaded.name, "input": in_str, "variant": vname,
                       "variant_info": VARIANT_INFO[vname], "params_M": params_m, "gmacs": gmacs}
                try:
                    v = Variant(vname, net)
                    row["weights_MB"] = v.weights_mb()
                    x1 = torch.randn(1, 3, *size, device=device, dtype=v.input_dtype)
                    if args.miou:
                        row["miou"], row["iou_per_class"], row["pixel_acc"] = eval_miou(
                            v, inputs, masks, num_labels)
                    row["latency_b1"] = measure_latency(v, x1, args.warmup, args.iters)
                    row["memory_b1_MB"] = None if "compile" in vname else measure_memory(v, x1)
                    try:
                        xb = torch.randn(args.batch, 3, *size, device=device, dtype=v.input_dtype)
                        row["throughput_fps"] = measure_throughput(v, xb)
                        row[f"memory_b{args.batch}_MB"] = None if "compile" in vname else measure_memory(v, xb)
                        del xb
                    except torch.cuda.OutOfMemoryError:
                        row["throughput_fps"] = None
                        print(f"  {vname}: batch {args.batch} does not fit in memory")
                    del x1, v
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                    print(f"  {vname}: ERROR {row['error']}")
                    rows.append(row)
                    torch.cuda.empty_cache()
                    continue
                torch.cuda.empty_cache()

                if base is None and vname == "fp32":
                    base = row
                if base is not None:
                    row["speedup"] = base["latency_b1"]["median_ms"] / row["latency_b1"]["median_ms"]
                    if "miou" in row and "miou" in base:
                        row["delta_miou"] = row["miou"] - base["miou"]
                lat = row["latency_b1"]
                print(f"  {vname:13s} lat {lat['median_ms']:7.2f} ms (p90 {lat['p90_ms']:.2f})"
                      f" | {fmt(row.get('throughput_fps'), '{:.1f}')} fps b={args.batch}"
                      f" | mem {fmt(row['memory_b1_MB'], '{:.0f}')} MB"
                      + (f" | mIoU {row['miou']:.4f}" if "miou" in row else ""))
                rows.append(row)
        del net, loaded
        torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, args.batch)
    meta = {"env": env, "args": {k: str(v) for k, v in vars(args).items()},
            "notes": "latency: batch 1, median over iters runs after warmup, CUDA events + synchronize "
                     "on every iteration; memory: peak allocated increase per forward (activations and "
                     "temporary buffers), weights reported separately; not measurable with CUDA Graphs"}
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    (OUT_DIR / f"{args.tag}.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nSaved: {OUT_DIR / (args.tag + '.json')}, {OUT_DIR / (args.tag + '.md')}")


if __name__ == "__main__":
    main()
