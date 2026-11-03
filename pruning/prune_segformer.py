import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_pruning as tp
from torch import nn
from transformers import AutoImageProcessor

PRUNING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PRUNING_DIR.parent))

import config as C 
from benchmark import eval_miou, load_test_set 
from camvid_dataset import build_label_mapping 
from dependency_groups import INPUT_SIZE, RESULTS, load_model 
from perf_models import SegFormerNet, Variant, count_gmacs, gpu_warmup, measure_latency 

TARGETS = ["mlp", "heads", "decoder", "residual4"]
IMPORTANCE = {"l1": lambda: tp.importance.GroupMagnitudeImportance(p=1),
              "random": lambda: tp.importance.RandomImportance()}


def blocks(model):
    return [blk for st in model.segformer.stages for blk in st.blocks]


def target_layers(model, target):
    if target == "mlp":
        return [l for blk in blocks(model) for l in (blk.mlp.fc1, blk.mlp.dwconv.dwconv)]
    if target == "heads":
        return [l for st in list(model.segformer.stages)[1:] for blk in st.blocks
                for l in (blk.attention.q_proj, blk.attention.k_proj, blk.attention.v_proj)]
    if target == "decoder":
        return [p.proj for p in model.decode_head.linear_projections] + [model.decode_head.linear_fuse]
    if target == "residual4":
        st = model.segformer.stages[3]
        return [st.patch_embeddings.proj] + [l for blk in st.blocks for l in (blk.attention.o_proj, blk.mlp.fc2)]
    raise ValueError(target)


def prune(model, target, ratio, importance):
    allowed = target_layers(model, target)
    ignored = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)) and m not in allowed]
    if target == "heads":
        kwargs = dict(num_heads={l: blk.attention.num_attention_heads for blk in blocks(model)
                                 for l in (blk.attention.q_proj, blk.attention.k_proj, blk.attention.v_proj)
                                 if l in allowed},
                      prune_num_heads=True, prune_head_dims=False, head_pruning_ratio=ratio)
    else:
        kwargs = dict(round_to=8)
    example = torch.randn(1, 3, *INPUT_SIZE)
    pruner = tp.pruner.MetaPruner(model, example_inputs={"pixel_values": example}, importance=IMPORTANCE[importance](),
                                  pruning_ratio=ratio, ignored_layers=ignored,
                                  output_transform=lambda out: out.logits, **kwargs)
    pruner.step()
    for blk in blocks(model):
        att = blk.attention
        att.num_attention_heads = att.q_proj.out_features // att.head_dim
    return model


def describe(model, target):
    if target == "mlp":
        return "Mix-FFN hidden " + "/".join(str(st.blocks[0].mlp.fc1.out_features) for st in model.segformer.stages)
    if target == "heads":
        return "heads " + "/".join(str(st.blocks[0].attention.num_attention_heads) for st in model.segformer.stages)
    if target == "decoder":
        dh = model.decode_head
        return (f"linear_c {'/'.join(str(p.proj.out_features) for p in dh.linear_projections)}, "
                f"width {dh.linear_fuse.out_channels}")
    if target == "residual4":
        return f"stage4 C {model.segformer.stages[3].patch_embeddings.proj.out_channels}"
    return ""


def measure(model, inputs, masks, num_labels, compile_latency):
    net = SegFormerNet(model).cuda().eval()
    row = {"params_M": sum(p.numel() for p in net.parameters()) / 1e6,
           "gmacs": count_gmacs(net, INPUT_SIZE)}
    fp32 = Variant("fp32", net)
    row["miou"] = eval_miou(fp32, inputs, masks, num_labels)[0]
    x = torch.randn(1, 3, *INPUT_SIZE, device="cuda")
    row["latency_fp32_ms"] = measure_latency(fp32, x)["median_ms"]
    if compile_latency:
        torch._dynamo.reset()
        row["latency_compile_ms"] = measure_latency(Variant("fp16+compile", net), x)["median_ms"]
    del net
    torch.cuda.empty_cache()
    return row


def fmt(v, f):
    return "—" if v is None else f.format(v)


def to_markdown(rows, base, split):
    head = ["Target", "Ratio", "Importance", "Shape after pruning", "Params, M", "GMACs",
            f"mIoU {split} (no fine-tuning)", "Δ mIoU", "Latency fp32, ms", "Latency fp16+compile, ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in [base] + rows:
        lines.append("| " + " | ".join([
            r["target"], f"{r['ratio']:.2f}", r["importance"], r["shape"],
            f"{r['params_M']:.3f}", f"{r['gmacs']:.2f}", f"{r['miou']:.4f}", f"{r['miou'] - base['miou']:+.4f}",
            f"{r['latency_fp32_ms']:.2f}", fmt(r.get("latency_compile_ms"), "{:.2f}"),
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--importance", nargs="+", default=["l1"], choices=list(IMPORTANCE))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--compile", action="store_true", help="also measure fp16+compile latency")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU required"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    mapping = build_label_mapping()
    processor = AutoImageProcessor.from_pretrained(str(C.PROJECT_DIR / "weights" / "segformer-b0-camvid"))
    images, masks = load_test_set(mapping, args.split)
    inputs = [processor(img, return_tensors="pt")["pixel_values"] for img in images]
    print(f"{args.split}: {len(inputs)} images")
    gpu_warmup()

    t0 = time.time()
    base = {"target": "baseline", "ratio": 0.0, "importance": "—", "shape": "original",
            **measure(load_model(), inputs, masks, mapping["num_labels"], args.compile)}
    print(f"baseline: params {base['params_M']:.3f}M, {base['gmacs']:.2f} GMACs, mIoU {base['miou']:.4f}, "
          f"fp32 {base['latency_fp32_ms']:.2f} ms ({time.time() - t0:.0f} s)")

    rows = []
    for target in args.targets:
        for importance in args.importance:
            for ratio in args.ratios:
                t0 = time.time()
                model = prune(load_model(), target, ratio, importance)
                row = {"target": target, "ratio": ratio, "importance": importance, "shape": describe(model, target),
                       **measure(model, inputs, masks, mapping["num_labels"], args.compile)}
                rows.append(row)
                print(f"{target:9s} {importance:6s} {ratio:.2f} | {row['shape']:34s} | params {row['params_M']:.3f}M "
                      f"| {row['gmacs']:.2f} GMACs | mIoU {row['miou']:.4f} ({row['miou'] - base['miou']:+.4f}) "
                      f"| fp32 {row['latency_fp32_ms']:.2f} ms"
                      + (f" | compile {row['latency_compile_ms']:.2f} ms" if "latency_compile_ms" in row else "")
                      + f" ({time.time() - t0:.0f} s)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, base, args.split)
    (RESULTS / f"prune_{args.tag}.md").write_text(md + "\n")
    (RESULTS / f"prune_{args.tag}.json").write_text(json.dumps({"args": vars(args), "baseline": base, "rows": rows},
                                                               indent=2))
    print("\n" + md)
    print(f"\nSaved: {RESULTS / ('prune_' + args.tag + '.md')}")


if __name__ == "__main__":
    main()
