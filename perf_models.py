import copy
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config as C

NATIVE_SIZE = tuple(C.NATIVE_SIZE)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

VARIANTS = ["fp32", "tf32", "fp16", "bf16", "half", "fp16+compile"]
VARIANT_INFO = {
    "fp32": "fp32, PyTorch defaults (cuDNN convolutions in TF32, matmul in fp32)",
    "tf32": "fp32 weights, TF32 also allowed for matmul (Ampere tensor cores)",
    "fp16": "torch.autocast(float16), fp32 weights",
    "bf16": "torch.autocast(bfloat16), fp32 weights",
    "half": "model.half(): fp16 weights and activations",
    "fp16+compile": "autocast(float16) + torch.compile(mode='reduce-overhead', CUDA Graphs)",
}


def parse_size(s):
    h, w = s.lower().split("x")
    return int(h), int(w)


class SegFormerNet(nn.Module):
    def __init__(self, hf_model, out_size=NATIVE_SIZE):
        super().__init__()
        self.m = hf_model
        self.out_size = out_size

    def forward(self, x):
        logits = self.m(pixel_values=x).logits
        return nn.functional.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)


class DeepLabNet(nn.Module):
    def __init__(self, seg_model, out_size=NATIVE_SIZE):
        super().__init__()
        self.m = seg_model
        self.out_size = out_size

    def forward(self, x):
        out = self.m(x)
        if tuple(out.shape[-2:]) != self.out_size:
            out = nn.functional.interpolate(out, size=self.out_size, mode="bilinear", align_corners=False)
        return out


class Loaded:
    def __init__(self, name, net, preprocess, parts):
        self.name = name
        self.net = net
        self.preprocess = preprocess
        self.parts = parts


def _segformer_stages(hf):
    sf = hf.segformer
    if hasattr(sf, "stages"):
        return [(st.patch_embeddings, st.blocks, st.layer_norm) for st in sf.stages]
    enc = sf.encoder
    return list(zip(enc.patch_embeddings, enc.block, enc.layer_norm))


def _sequence_reduction(attention):
    if hasattr(attention, "q_proj"):
        sr = getattr(attention, "sequence_reduction", None)
        return None if sr is None else (sr, sr)
    att = attention.self
    return (att.sr, att.layer_norm) if getattr(att, "sr_ratio", 1) > 1 else None


def load_segformer(model_dir):
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    hf = AutoModelForSemanticSegmentation.from_pretrained(str(model_dir)).eval()

    def preprocess(image, size):
        enc = processor(image, size={"height": size[0], "width": size[1]}, return_tensors="pt")
        return enc["pixel_values"]

    parts = []
    for i, (patch_embed, blocks, norm) in enumerate(_segformer_stages(hf)):
        s = f"stage{i + 1}"
        parts.append((s, patch_embed, norm, 0))
        parts.append((f"{s}.patch_embed", patch_embed, patch_embed, 1))
        for layer in blocks:
            parts.append((f"{s}.attention", layer.attention, layer.attention, 1))
            sr = _sequence_reduction(layer.attention)
            if sr is not None:
                parts.append((f"{s}.attention.sr_conv+norm", sr[0], sr[1], 2))
            parts.append((f"{s}.mix_ffn", layer.mlp, layer.mlp, 1))
    dh = hf.decode_head
    parts.append(("decode_head", dh, dh, 0))
    for i, mlp in enumerate(getattr(dh, "linear_projections", None) or dh.linear_c):
        parts.append((f"decode_head.linear_c{i + 1}", mlp, mlp, 1))
    parts.append(("decode_head.linear_fuse", dh.linear_fuse, dh.linear_fuse, 1))
    parts.append(("decode_head.classifier", dh.classifier, dh.classifier, 1))
    return Loaded("SegFormer-B0", SegFormerNet(hf), preprocess, parts)


def load_deeplab(ckpt, repo):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.append(repo)
    from seg_utils import load_model as load_deeplab_ckpt

    seg = load_deeplab_ckpt(str(ckpt), "cpu").eval()

    def preprocess(image, size):
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        if tuple(x.shape[-2:]) != tuple(size):
            x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

    net = seg.net
    bb = net.backbone
    parts = [("stem (conv1+bn+relu+maxpool)", bb.conv1, bb.maxpool, 0)]
    for i in range(1, 5):
        parts.append((f"layer{i}", getattr(bb, f"layer{i}"), getattr(bb, f"layer{i}"), 0))
    parts.append(("ASPP", net.classifier[0], net.classifier[0], 0))
    parts.append(("head (conv3x3+bn+relu+conv1x1)", net.classifier[1], net.classifier[4], 0))
    return Loaded("DeepLabV3-ResNet50", DeepLabNet(seg), preprocess, parts)


def load(name, args):
    if name == "segformer":
        return load_segformer(args.segformer)
    if name == "deeplab":
        return load_deeplab(args.deeplab_ckpt, args.deeplab_repo)
    raise ValueError(name)


def add_model_args(ap):
    ap.add_argument("--segformer", type=Path, default=None,
                    help="SegFormer weights folder; default weights/segformer-b0-camvid, "
                         "falls back to config.OUTPUT_DIR")
    ap.add_argument("--deeplab-repo", type=Path, default=C.PROJECT_DIR.parent / "Fine_Tuning_DeepLabV3_ResNet50",
                    help="DeepLabV3 project folder (provides seg_utils.py and camvid.py)")
    ap.add_argument("--deeplab-ckpt", type=Path, default=None,
                    help="default <deeplab-repo>/runs/full_ce_lovasz_e50/best.pt")


def resolve_model_args(args):
    if getattr(args, "segformer", None) is None:
        repo_weights = C.PROJECT_DIR / "weights" / "segformer-b0-camvid"
        args.segformer = repo_weights if repo_weights.is_dir() else C.OUTPUT_DIR
    if getattr(args, "deeplab_ckpt", None) is None:
        args.deeplab_ckpt = args.deeplab_repo / "runs" / "full_ce_lovasz_e50" / "best.pt"
    return args


class Variant:
    def __init__(self, name, net):
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name}, expected one of {VARIANTS}")
        if name == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is not supported on this GPU")
        self.name = name
        self.input_dtype = torch.float16 if name == "half" else torch.float32
        self.autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                               "fp16+compile": torch.float16}.get(name)
        self._module_map = {}
        if name in ("half", "fp16+compile"):
            orig = net
            net = copy.deepcopy(net)
            self._module_map = {id(a): b for a, b in zip(orig.modules(), net.modules())}
        if name == "half":
            net = net.half()
        net = net.eval()
        self.module = net
        self.net = torch.compile(net, mode="reduce-overhead") if name == "fp16+compile" else net

    def map_module(self, m):
        return self._module_map.get(id(m), m)

    def set_backend_flags(self):
        torch.backends.cuda.matmul.allow_tf32 = self.name == "tf32"
        torch.backends.cudnn.allow_tf32 = True

    def __call__(self, x):
        self.set_backend_flags()
        if x.dtype != self.input_dtype:
            x = x.to(self.input_dtype)
        if self.autocast_dtype is not None:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                return self.net(x)
        return self.net(x)

    def weights_mb(self):
        n = sum(p.numel() * p.element_size() for p in self.module.parameters())
        n += sum(b.numel() * b.element_size() for b in self.module.buffers())
        return n / 2 ** 20


def gpu_warmup(seconds=3.0):
    a = torch.randn(2048, 2048, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = (a @ a).clamp_(-1, 1)
    torch.cuda.synchronize()


@torch.inference_mode()
def measure_latency(fn, x, warmup=30, iters=200):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = np.asarray(times)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "min_ms": float(t.min()), "std_ms": float(t.std())}


@torch.inference_mode()
def measure_throughput(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return x.shape[0] * iters / (start.elapsed_time(end) / 1000.0)


@torch.inference_mode()
def measure_memory(fn, x):
    fn(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(x)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


@torch.no_grad()
def count_gmacs(net, size, device="cuda"):
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 3, *size, device=device)
    with FlopCounterMode(display=False) as fc:
        net(x)
    return fc.get_total_flops() / 2 / 1e9


def count_params(net):
    return sum(p.numel() for p in net.parameters())


@torch.inference_mode()
def part_breakdown(fn, x, parts, iters=50, warmup=10):
    stacks, pairs, handles = {}, {}, []

    def pre_hook(name):
        def hook(module, inputs):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            stacks.setdefault(name, []).append(ev)
        return hook

    def post_hook(name):
        def hook(module, inputs, output):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            pairs.setdefault(name, []).append((stacks[name].pop(), ev))
        return hook

    for name, first, last, _ in parts:
        handles.append(first.register_forward_pre_hook(pre_hook(name)))
        handles.append(last.register_forward_hook(post_hook(name)))
    try:
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        pairs.clear()
        totals = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(x)
            e.record()
            totals.append((s, e))
        torch.cuda.synchronize()
        res = {name: sum(a.elapsed_time(b) for a, b in lst) / iters for name, lst in pairs.items()}
        res["total"] = sum(a.elapsed_time(b) for a, b in totals) / iters
        return res
    finally:
        for h in handles:
            h.remove()


def env_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_GB=round(p.total_memory / 2 ** 30, 1))
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info
