import sys
from collections import Counter
from pathlib import Path

import torch
import torch_pruning as tp
from transformers import AutoModelForSemanticSegmentation

PRUNING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PRUNING_DIR.parent))

import config as C  

WEIGHTS = C.PROJECT_DIR / "weights" / "segformer-b0-camvid"
RESULTS = PRUNING_DIR / "results"
INPUT_SIZE = C.NATIVE_SIZE


def load_model():
    return AutoModelForSemanticSegmentation.from_pretrained(str(WEIGHTS)).eval()


def build_graph(model, example):
    return tp.DependencyGraph().build_dependency(
        model, example_inputs={"pixel_values": example}, output_transform=lambda out: out.logits
    )


def short_name(name):
    return name.replace("segformer.stages.", "s").replace("decode_head.", "dh.")


def group_layers(group, names):
    layers = []
    for dep, _ in group:
        module = dep.target.module
        if module in names:
            side = "out" if "out_channels" in dep.handler.__name__ else "in"
            layers.append((names[module], side))
    return list(dict.fromkeys(layers))


def classify(layers):
    names = [n for n, _ in layers]
    if any(n.endswith("layernorm_before") for n in names):
        return "residual stream C"
    if any(n.endswith("attention.q_proj") for n, side in layers if side == "out"):
        return "attention Q/K/V -> O"
    if any("sequence_reduction" in n for n, side in layers if side == "out"):
        return "key/value reduction"
    if any(n.endswith("mlp.fc1") for n, side in layers if side == "out"):
        return "Mix-FFN hidden"
    if any(n.endswith("linear_fuse") for n, side in layers if side == "out"):
        return "decoder fuse"
    if any("linear_projections" in n for n, side in layers if side == "out"):
        return "decoder projection"
    return "other"


def stage_of(layers):
    for n, _ in layers:
        if n.startswith("segformer.stages."):
            return f"stage{int(n.split('.')[2]) + 1}"
    return "decoder"


def collect_groups(model, graph):
    names = {m: n for n, m in model.named_modules()}
    groups = []
    for g in graph.get_all_groups(ignored_layers=[model.decode_head.classifier]):
        layers = group_layers(g, names)
        groups.append({"type": classify(layers), "stage": stage_of(layers),
                       "channels": len(g[0][1]), "layers": layers})
    return groups


def head_alignment_demo(example):
    rows = []
    cases = [("1 channel", [10]),
             ("32 channels, not aligned to a head", list(range(10, 42))),
             ("32 channels = whole head 1", list(range(32, 64)))]
    for title, idxs in cases:
        model = load_model()
        with torch.no_grad():
            ref = model(pixel_values=example).logits
        graph = build_graph(model, example)
        att = model.segformer.stages[1].blocks[0].attention
        graph.get_pruning_group(att.q_proj, tp.prune_linear_out_channels, idxs=idxs).prune()
        try:
            with torch.no_grad():
                out = model(pixel_values=example).logits
            result = f"forward OK, max abs logit diff = {float((out - ref).abs().max()):.2f}"
        except RuntimeError as e:
            result = f"forward fails: {str(e).splitlines()[0][:90]}"
        rows.append((title, tuple(att.q_proj.weight.shape), result))
    return rows


def to_markdown(groups, demo):
    lines = ["# SegFormer-B0 dependency groups (torch_pruning DepGraph)", "",
             f"Input {INPUT_SIZE[0]}x{INPUT_SIZE[1]}, classifier ignored. Total groups: {len(groups)}.", "",
             "## Summary by type", "", "| Type | Groups | Channels per group | Layers per group |",
             "|---|---|---|---|"]
    by_type = {}
    for g in groups:
        by_type.setdefault(g["type"], []).append(g)
    for t, gs in by_type.items():
        ch = sorted({g["channels"] for g in gs})
        ly = sorted({len(g["layers"]) for g in gs})
        lines.append(f"| {t} | {len(gs)} | {', '.join(map(str, ch))} | {', '.join(map(str, ly))} |")

    lines += ["", "## All groups", "", "| # | Type | Stage | Channels | Layers |", "|---|---|---|---|---|"]
    for i, g in enumerate(groups, 1):
        lines.append(f"| {i} | {g['type']} | {g['stage']} | {g['channels']} | {len(g['layers'])} |")

    lines += ["", "## Coupled layers, one example per type", ""]
    seen = set()
    for g in groups:
        if g["type"] in seen:
            continue
        seen.add(g["type"])
        lines += [f"### {g['type']} ({g['stage']}, {g['channels']} channels)", ""]
        lines += [f"- `{short_name(n)}` ({side} channels)" for n, side in g["layers"]]
        lines.append("")

    lines += ["## Pitfall: attention heads", "",
              "Pruning one group in `stage2.blocks.0.attention` (2 heads x head_dim 32 = 64 channels):", "",
              "| Removed | q_proj weight after | Result |", "|---|---|---|"]
    lines += [f"| {t} | {s} | {r} |" for t, s, r in demo]
    return "\n".join(lines) + "\n"


def main():
    torch.manual_seed(0)
    example = torch.randn(1, 3, *INPUT_SIZE)
    model = load_model()
    graph = build_graph(model, example)
    groups = collect_groups(model, graph)

    counts = Counter(g["type"] for g in groups)
    print(f"{len(groups)} dependency groups:")
    for t, n in counts.items():
        print(f"  {t:22s} {n}")

    demo = head_alignment_demo(example)
    print("\nattention head pitfall (stage2.blocks.0):")
    for t, s, r in demo:
        print(f"  {t:36s} q_proj {s} -> {r}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "dependency_groups.md"
    out.write_text(to_markdown(groups, demo))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
