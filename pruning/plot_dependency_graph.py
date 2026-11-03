import json
import re

import torch
from torch import nn

from dependency_groups import INPUT_SIZE, RESULTS, build_graph, collect_groups, load_model

COLORS = {
    "residual stream C": "#5b7bd5",
    "attention Q/K/V -> O": "#e8913a",
    "key/value reduction": "#9b6bd6",
    "Mix-FFN hidden": "#3fa66b",
    "decoder projection": "#2fa7b5",
    "decoder fuse": "#d0505f",
    "not pruned": "#9aa0a6",
}

OP_NAMES = {"Add": "add", "Transpose": "transpose", "Permute": "permute", "Gelu": "gelu", "Relu": "relu",
            "UpsampleBilinear2D": "upsample", "Cat": "cat", "T": "t", "Addmm": "addmm", "Mm": "mm",
            "Bmm": "bmm", "Mul": "mul", "Div": "div", "Clone": "clone", "Flatten": "flatten"}


def stage_name(name):
    if name.startswith("segformer.stages."):
        return f"stage{int(name.split('.')[2]) + 1}"
    return "decoder"


def compact(name):
    name = re.sub(r"segformer\.stages\.(\d)", lambda m: f"s{int(m.group(1)) + 1}", name)
    return (name.replace("decode_head.", "dec.").replace("blocks.", "b").replace("attention.", "attn.")
            .replace("patch_embeddings.", "patch.").replace("sequence_reduction.sequence_reduction", "sr.conv")
            .replace("sequence_reduction.layer_norm", "sr.norm").replace("linear_projections.", "linear_c")
            .replace("dwconv.dwconv", "dwconv"))


def shape_of(module):
    if isinstance(module, nn.Linear):
        return f"{module.in_features} -> {module.out_features}"
    if isinstance(module, nn.Conv2d):
        k = "x".join(map(str, module.kernel_size))
        g = f", groups {module.groups}" if module.groups > 1 else ""
        return f"{module.in_channels} -> {module.out_channels}, {k}{g}"
    if isinstance(module, nn.LayerNorm):
        return f"{module.normalized_shape[0]}"
    if isinstance(module, nn.BatchNorm2d):
        return f"{module.num_features}"
    return ""


def op_label(node):
    m = re.search(r"\((\w+?)Backward\d*\)", node.name)
    if m:
        base = m.group(1)
        if "ScaledDotProduct" in base:
            return "SDPA"
        return OP_NAMES.get(base, base.lower())
    if "Reshape" in node.name:
        return "reshape"
    if "Concat" in node.name:
        return "cat"
    return node.name.split("_")[1].lower()


def layer_groups(groups):
    out_type, in_types = {}, {}
    for g in groups:
        for name, side in g["layers"]:
            if side == "out":
                out_type[name] = g["type"]
            else:
                in_types.setdefault(name, []).append(g["type"])
    return out_type, in_types


def build_data(model, graph, groups):
    names = {m: n for n, m in model.named_modules()}
    out_type, in_types = layer_groups(groups)
    nodes = list(graph.module2node.values())
    ids = {n: i for i, n in enumerate(nodes)}
    is_layer = {n: n.module in names for n in nodes}

    stage = {n: stage_name(names[n.module]) for n in nodes if is_layer[n]}
    for _ in range(len(nodes)):
        changed = False
        for n in nodes:
            if n in stage:
                continue
            src = [stage[i] for i in n.inputs if i in stage] or [stage[o] for o in n.outputs if o in stage]
            if src:
                stage[n] = src[0]
                changed = True
        if not changed:
            break

    js_nodes = []
    for n in nodes:
        if is_layer[n]:
            name = names[n.module]
            gtype = out_type.get(name, "not pruned")
            ins = ", ".join(sorted(set(in_types.get(name, [])))) or "—"
            js_nodes.append({"id": ids[n], "label": f"{compact(name)}\n{shape_of(n.module)}", "layer": True,
                             "stage": stage[n], "group": gtype, "color": COLORS[gtype],
                             "title": f"{name}\n{type(n.module).__name__}: {shape_of(n.module)}\n"
                                      f"out channels -> group: {gtype}\nin channels -> group: {ins}"})
        else:
            js_nodes.append({"id": ids[n], "label": op_label(n), "layer": False, "stage": stage.get(n, "decoder"),
                             "group": "op", "color": "#d9dce1", "title": n.name})

    full_edges = [{"from": ids[n], "to": ids[o]} for n in nodes for o in n.outputs]

    layer_edges = set()
    for n in nodes:
        if not is_layer[n]:
            continue
        stack, seen = list(n.outputs), set()
        while stack:
            o = stack.pop()
            if o in seen:
                continue
            seen.add(o)
            if is_layer[o]:
                layer_edges.add((ids[n], ids[o]))
            else:
                stack.extend(o.outputs)
    layer_edges = [{"from": a, "to": b} for a, b in sorted(layer_edges)]
    return {"nodes": js_nodes, "full_edges": full_edges, "layer_edges": layer_edges, "colors": COLORS}


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SegFormer Dependency Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root { --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --panel: #f6f8fa; --border: #d0d7de; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22; --border: #30363d; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--panel);
           display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; }
  h1 { font-size: 16px; margin: 0; }
  label { color: var(--muted); }
  select { font: inherit; padding: 2px 6px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  #graph { position: absolute; top: 0; bottom: 0; left: 0; right: 0; }
  main { position: relative; height: calc(100vh - 110px); min-height: 400px; }
  #info { padding: 6px 16px; color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>SegFormer-B0 dependency graph</h1>
  <label>View <select id="view"><option value="layers">layers only</option><option value="full">full graph (with ops)</option></select></label>
  <label>Part <select id="stage"><option value="stage1">stage1</option><option value="stage2">stage2</option>
    <option value="stage3">stage3</option><option value="stage4">stage4</option><option value="decoder">decoder</option>
    <option value="all">whole model</option></select></label>
  <div class="legend" id="legend"></div>
</header>
<div id="info">Node color = dependency group of the layer's output channels. Hover a node for its shape and groups; drag to pan, scroll to zoom.</div>
<main><div id="graph"></div></main>
<script>
const DATA = __DATA__;
const legend = document.getElementById("legend");
for (const [name, color] of Object.entries(DATA.colors)) {
  legend.insertAdjacentHTML("beforeend", `<span><i style="background:${color}"></i>${name}</span>`);
}
legend.insertAdjacentHTML("beforeend", `<span><i style="background:#d9dce1"></i>op without weights</span>`);
const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const container = document.getElementById("graph");
let network = null;
function render() {
  const view = document.getElementById("view").value;
  const stage = document.getElementById("stage").value;
  const keep = DATA.nodes.filter(n => (view === "full" || n.layer) && (stage === "all" || n.stage === stage));
  const ids = new Set(keep.map(n => n.id));
  const edges = (view === "full" ? DATA.full_edges : DATA.layer_edges).filter(e => ids.has(e.from) && ids.has(e.to));
  const nodes = keep.map(n => ({
    id: n.id, label: n.label, title: n.title,
    shape: n.layer ? "box" : "ellipse",
    color: { background: n.color, border: n.color },
    font: { color: n.layer ? "#ffffff" : "#1f2328", size: n.layer ? 13 : 10, face: "monospace" },
  }));
  const options = {
    layout: { hierarchical: { direction: "UD", sortMethod: "directed", levelSeparation: 70, nodeSpacing: 170 } },
    physics: false,
    edges: { arrows: "to", color: { color: dark ? "#6e7681" : "#8c959f" }, smooth: { type: "cubicBezier" } },
    interaction: { hover: true, tooltipDelay: 100 },
  };
  if (network) network.destroy();
  network = new vis.Network(container, { nodes, edges }, options);
  document.getElementById("info").textContent =
    `${nodes.length} nodes, ${edges.length} edges. Node color = dependency group of the layer's output channels. ` +
    `Hover a node for its shape and groups; drag to pan, scroll to zoom.`;
}
document.getElementById("view").addEventListener("change", render);
document.getElementById("stage").addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def main():
    torch.manual_seed(0)
    model = load_model()
    graph = build_graph(model, torch.randn(1, 3, *INPUT_SIZE))
    data = build_data(model, graph, collect_groups(model, graph))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "dependency_graph.html"
    out.write_text(HTML.replace("__DATA__", json.dumps(data)))
    n_layers = sum(n["layer"] for n in data["nodes"])
    print(f"{len(data['nodes'])} nodes ({n_layers} layers), {len(data['full_edges'])} edges, "
          f"{len(data['layer_edges'])} layer-to-layer edges")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
