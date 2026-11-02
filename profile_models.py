import argparse
import json
import re
from collections import OrderedDict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import config as C
from perf_models import (Variant, add_model_args, env_info, gpu_warmup, load, measure_latency,
                         parse_size, part_breakdown, resolve_model_args)

OUT_ROOT = C.PROJECT_DIR / "results" / "profile"
ANNOTATION = "forward"

CATEGORIES = OrderedDict([
    ("attention (SDPA)", r"aten::(_scaled_dot_product.*|_efficient_attention.*|_flash_attention.*)"),
    ("matmul / linear (addmm, mm, bmm)", r"aten::(addmm|mm|bmm|baddbmm|matmul|linear)"),
    ("convolution", r"aten::(.*conv.*|_convolution|cudnn_convolution.*|thnn_conv.*)"),
    ("softmax", r"aten::(_?softmax|_softmax_backward.*)"),
    ("normalization (LayerNorm, BatchNorm)", r"aten::(native_layer_norm|layer_norm|.*batch_norm.*)"),
    ("activation (GELU, ReLU)", r"aten::(gelu|relu_?|clamp_min_?|silu|hardtanh_?)"),
    ("interpolate / pooling", r"aten::(upsample.*|_upsample.*|.*pool.*|adaptive.*)"),
    ("copy / layout (contiguous, cat, cast)", r"aten::(copy_|contiguous|clone|cat|_to_copy|to|"
                                              r"index.*|transpose|permute|reshape|view|_unsafe_view)"),
    ("elementwise (add, mul, div)", r"aten::(add_?|mul_?|div_?|sub_?|rsub|where|fill_|zero_)"),
])


def dev_time(evt, self_time=True):
    names = (["self_device_time_total", "self_cuda_time_total"] if self_time
             else ["device_time_total", "cuda_time_total"])
    for name in names:
        if hasattr(evt, name):
            return getattr(evt, name)
    return 0.0


def is_kernel(evt):
    return str(getattr(evt, "device_type", "")).upper().endswith("CUDA") and evt.key != ANNOTATION


def table(prof_avg, rows):
    for key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            return prof_avg.table(sort_by=key, row_limit=rows)
        except (AttributeError, KeyError, ValueError):
            continue
    return prof_avg.table(row_limit=rows)


def categorize(avg, iters):
    cats = OrderedDict((c, 0.0) for c in CATEGORIES)
    cats["other"] = 0.0
    other_ops = {}
    for e in avg:
        if is_kernel(e) or not e.key.startswith("aten::"):
            continue
        t = dev_time(e) / iters
        if t <= 0:
            continue
        for cat, pat in CATEGORIES.items():
            if re.fullmatch(pat, e.key):
                cats[cat] += t
                break
        else:
            cats["other"] += t
            other_ops[e.key] = other_ops.get(e.key, 0.0) + t
    return cats, dict(sorted(other_ops.items(), key=lambda kv: -kv[1])[:10])


def top_ops(avg, iters, n=15):
    ops = [e for e in avg if not is_kernel(e) and e.key.startswith("aten::") and dev_time(e) > 0]
    ops.sort(key=lambda e: -dev_time(e))
    total = sum(dev_time(e) for e in ops)
    return [{"op": e.key, "calls_per_forward": e.count / iters,
             "self_gpu_us_per_forward": dev_time(e) / iters,
             "share": dev_time(e) / total if total else 0.0} for e in ops[:n]]


def to_markdown(s):
    lines = [f"# Profile: {s['model']}, input {s['input']}, {s['variant']}, batch {s['batch']}", ""]
    lines += [
        "| Metric | Value |", "|---|---|",
        f"| latency without profiler, median | {s['latency_ms']:.2f} ms |",
        f"| total GPU kernel time per forward | {s['kernel_ms_per_forward']:.2f} ms |",
        f"| GPU busy = kernels / latency | {s['gpu_busy']:.0%} -> {s['verdict']} |",
        f"| CUDA kernel launches per forward | {s['kernels_per_forward']:.0f} |",
        f"| average kernel | {s['avg_kernel_us']:.1f} us |",
        f"| CPU time of aten ops per forward (with profiler overhead) | {s['cpu_ms_per_forward']:.2f} ms |",
        f"| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | {s['cuda_api_ms_per_forward']:.2f} ms |",
        "", "## Most expensive ops (self GPU time)", "",
        "| Op | Calls per forward | us per forward | Share |", "|---|---|---|---|",
    ]
    for o in s["top_ops"]:
        lines.append(f"| `{o['op']}` | {o['calls_per_forward']:.0f} | "
                     f"{o['self_gpu_us_per_forward']:.0f} | {o['share']:.1%} |")
    lines += ["", "## By category", "", "| Category | us per forward | Share |", "|---|---|---|"]
    tot = sum(s["categories_us"].values()) or 1
    for c, t in s["categories_us"].items():
        if t > 0:
            lines.append(f"| {c} | {t:.0f} | {t / tot:.1%} |")
    lines += ["", "## By model part (CUDA events in hooks)", "",
              "| Part | ms per forward | Share of forward |", "|---|---|---|"]
    total = s["parts_ms"]["total"]
    for name, level in s["parts_order"]:
        if name in s["parts_ms"]:
            t = s["parts_ms"][name]
            lines.append(f"| {'&nbsp;&nbsp;&nbsp;&nbsp;' * level}{name} | {t:.3f} | {t / total:.1%} |")
    lines.append(f"| other (outside top-level parts: final upsample etc.) | "
                 f"{s['parts_ms']['other_top_level']:.3f} | {s['parts_ms']['other_top_level'] / total:.1%} |")
    lines.append(f"| **whole forward** | **{total:.3f}** | 100% |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--model", required=True, choices=["segformer", "deeplab"])
    ap.add_argument("--size", default="360x480")
    ap.add_argument("--variant", default="fp32", choices=["fp32", "tf32", "fp16", "bf16", "half"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20, help="forward passes under the profiler")
    args = resolve_model_args(ap.parse_args())

    assert torch.cuda.is_available(), "GPU required"
    torch.backends.cudnn.benchmark = True
    size = parse_size(args.size)
    loaded = load(args.model, args)
    net = loaded.net.cuda().eval()
    v = Variant(args.variant, net)
    x = torch.randn(args.batch, 3, *size, device="cuda", dtype=v.input_dtype)
    out_dir = OUT_ROOT / f"{args.model}_{size[0]}x{size[1]}_{args.variant}_b{args.batch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_warmup()
    lat = measure_latency(v, x)

    with torch.inference_mode():
        for _ in range(10):
            v(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(args.iters):
                with record_function(ANNOTATION):
                    v(x)
            torch.cuda.synchronize()

    avg = prof.key_averages()
    (out_dir / "ops_by_self_gpu_time.txt").write_text(table(avg, 30))
    (out_dir / "ops_by_input_shape.txt").write_text(table(prof.key_averages(group_by_input_shape=True), 40))
    prof.export_chrome_trace(str(out_dir / "trace.json"))

    kernels = [e for e in avg if is_kernel(e)]
    kernel_us = sum(dev_time(e) for e in kernels) / args.iters
    n_kernels = sum(e.count for e in kernels) / args.iters
    cpu_us = sum(e.self_cpu_time_total for e in avg
                 if not is_kernel(e) and e.key.startswith("aten::")) / args.iters
    cuda_api_us = sum(e.self_cpu_time_total for e in avg
                      if not is_kernel(e) and e.key.startswith("cuda")) / args.iters
    busy = kernel_us / 1000.0 / lat["median_ms"]
    verdict = ("GPU-bound: kernels run almost back to back" if busy > 0.85 else
               "partly launch-bound: noticeable GPU idle gaps" if busy > 0.6 else
               "launch-bound (CPU-bound): GPU mostly waits for kernel launches")
    cats, other_ops = categorize(avg, args.iters)

    parts = part_breakdown(v, x, [(n, v.map_module(a), v.map_module(b), lvl) for n, a, b, lvl in loaded.parts])
    top_total = sum(parts.get(name, 0.0) for name, _, _, level in loaded.parts if level == 0)
    parts["other_top_level"] = parts["total"] - top_total
    order = list(OrderedDict(((name, level), None) for name, _, _, level in loaded.parts))

    summary = {
        "model": loaded.name, "input": f"{size[0]}x{size[1]}", "variant": args.variant,
        "batch": args.batch, "env": env_info(),
        "latency_ms": lat["median_ms"], "latency": lat,
        "kernel_ms_per_forward": kernel_us / 1000.0, "gpu_busy": busy, "verdict": verdict,
        "kernels_per_forward": n_kernels, "avg_kernel_us": kernel_us / max(n_kernels, 1),
        "cpu_ms_per_forward": cpu_us / 1000.0, "cuda_api_ms_per_forward": cuda_api_us / 1000.0,
        "top_ops": top_ops(avg, args.iters), "categories_us": cats, "other_ops_us": other_ops,
        "parts_ms": parts, "parts_order": order,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (out_dir / "summary.md").write_text(md)
    print(table(avg, 20))
    print(md)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
