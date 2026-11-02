# Profile: SegFormer-B0, input 360x480, tf32, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 7.09 ms |
| total GPU kernel time per forward | 7.00 ms |
| GPU busy = kernels / latency | 99% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 235 |
| average kernel | 29.8 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.47 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 1.17 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 12 | 1148 | 16.4% |
| `aten::_efficient_attention_forward` | 8 | 877 | 12.5% |
| `aten::native_layer_norm` | 30 | 860 | 12.3% |
| `aten::copy_` | 29 | 807 | 11.5% |
| `aten::addmm` | 40 | 596 | 8.5% |
| `aten::upsample_bilinear2d` | 5 | 578 | 8.3% |
| `aten::mm` | 12 | 452 | 6.5% |
| `aten::_conv_depthwise2d` | 8 | 448 | 6.4% |
| `aten::cat` | 1 | 430 | 6.1% |
| `aten::add_` | 23 | 227 | 3.2% |
| `aten::gelu` | 8 | 211 | 3.0% |
| `aten::add` | 16 | 138 | 2.0% |
| `aten::cudnn_batch_norm` | 1 | 112 | 1.6% |
| `aten::clamp_min` | 1 | 108 | 1.5% |
| `aten::fill_` | 16 | 14 | 0.2% |

## By category

| Category | us per forward | Share |
|---|---|---|
| attention (SDPA) | 877 | 12.5% |
| matmul / linear (addmm, mm, bmm) | 1048 | 15.0% |
| convolution | 1596 | 22.8% |
| normalization (LayerNorm, BatchNorm) | 972 | 13.9% |
| activation (GELU, ReLU) | 318 | 4.5% |
| interpolate / pooling | 578 | 8.3% |
| copy / layout (contiguous, cat, cast) | 1237 | 17.7% |
| elementwise (add, mul, div) | 378 | 5.4% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stage1 | 2.275 | 30.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.patch_embed | 0.176 | 2.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.attention | 0.765 | 10.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage1.attention.sr_conv+norm | 0.178 | 2.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.mix_ffn | 0.730 | 9.9% |
| stage2 | 1.039 | 14.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.patch_embed | 0.082 | 1.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.attention | 0.400 | 5.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage2.attention.sr_conv+norm | 0.088 | 1.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.mix_ffn | 0.384 | 5.2% |
| stage3 | 0.833 | 11.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.patch_embed | 0.059 | 0.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.attention | 0.361 | 4.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage3.attention.sr_conv+norm | 0.090 | 1.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.mix_ffn | 0.333 | 4.5% |
| stage4 | 0.502 | 6.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.patch_embed | 0.065 | 0.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.attention | 0.175 | 2.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.mix_ffn | 0.204 | 2.8% |
| decode_head | 2.629 | 35.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c1 | 0.198 | 2.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c2 | 0.065 | 0.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c3 | 0.024 | 0.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c4 | 0.013 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_fuse | 0.748 | 10.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.classifier | 0.076 | 1.0% |
| other (outside top-level parts: final upsample etc.) | 0.109 | 1.5% |
| **whole forward** | **7.388** | 100% |
