# Profile: SegFormer-B0, input 360x480, fp16, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 7.92 ms |
| total GPU kernel time per forward | 5.51 ms |
| GPU busy = kernels / latency | 70% -> partly launch-bound: noticeable GPU idle gaps |
| CUDA kernel launches per forward | 435 |
| average kernel | 12.7 us |
| CPU time of aten ops per forward (with profiler overhead) | 7.10 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 2.01 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::copy_` | 234 | 1609 | 29.2% |
| `aten::native_layer_norm` | 30 | 864 | 15.7% |
| `aten::cudnn_convolution` | 20 | 767 | 13.9% |
| `aten::upsample_bilinear2d` | 10 | 578 | 10.5% |
| `aten::cat` | 1 | 430 | 7.8% |
| `aten::addmm` | 40 | 274 | 5.0% |
| `aten::add_` | 31 | 262 | 4.8% |
| `aten::mm` | 12 | 213 | 3.9% |
| `aten::_flash_attention_forward` | 8 | 161 | 2.9% |
| `aten::add` | 16 | 136 | 2.5% |
| `aten::gelu` | 8 | 97 | 1.8% |
| `aten::cudnn_batch_norm` | 1 | 63 | 1.2% |
| `aten::clamp_min` | 1 | 54 | 1.0% |

## By category

| Category | us per forward | Share |
|---|---|---|
| attention (SDPA) | 161 | 2.9% |
| matmul / linear (addmm, mm, bmm) | 487 | 8.8% |
| convolution | 767 | 13.9% |
| normalization (LayerNorm, BatchNorm) | 928 | 16.8% |
| activation (GELU, ReLU) | 151 | 2.7% |
| interpolate / pooling | 578 | 10.5% |
| copy / layout (contiguous, cat, cast) | 2039 | 37.0% |
| elementwise (add, mul, div) | 397 | 7.2% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stage1 | 1.620 | 17.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.patch_embed | 0.200 | 2.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.attention | 0.351 | 3.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage1.attention.sr_conv+norm | 0.185 | 1.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.mix_ffn | 0.474 | 5.0% |
| stage2 | 1.305 | 13.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.patch_embed | 0.083 | 0.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.attention | 0.556 | 5.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage2.attention.sr_conv+norm | 0.192 | 2.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.mix_ffn | 0.404 | 4.2% |
| stage3 | 2.148 | 22.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.patch_embed | 0.172 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.attention | 1.088 | 11.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage3.attention.sr_conv+norm | 0.363 | 3.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.mix_ffn | 0.525 | 5.5% |
| stage4 | 1.704 | 17.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.patch_embed | 0.170 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.attention | 0.627 | 6.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.mix_ffn | 0.547 | 5.7% |
| decode_head | 2.485 | 26.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c1 | 0.162 | 1.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c2 | 0.043 | 0.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c3 | 0.027 | 0.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c4 | 0.020 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_fuse | 0.652 | 6.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.classifier | 0.044 | 0.5% |
| other (outside top-level parts: final upsample etc.) | 0.269 | 2.8% |
| **whole forward** | **9.530** | 100% |
