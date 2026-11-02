# Profile: SegFormer-B0, input 360x480, fp32, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 7.38 ms |
| total GPU kernel time per forward | 7.33 ms |
| GPU busy = kernels / latency | 99% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 252 |
| average kernel | 29.1 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.39 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 1.47 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 12 | 1145 | 15.6% |
| `aten::_efficient_attention_forward` | 8 | 878 | 12.0% |
| `aten::addmm` | 40 | 863 | 11.8% |
| `aten::native_layer_norm` | 30 | 858 | 11.7% |
| `aten::copy_` | 29 | 825 | 11.3% |
| `aten::upsample_bilinear2d` | 5 | 578 | 7.9% |
| `aten::mm` | 12 | 479 | 6.5% |
| `aten::_conv_depthwise2d` | 8 | 448 | 6.1% |
| `aten::cat` | 1 | 430 | 5.9% |
| `aten::add_` | 23 | 236 | 3.2% |
| `aten::gelu` | 8 | 209 | 2.8% |
| `aten::add` | 16 | 147 | 2.0% |
| `aten::cudnn_batch_norm` | 1 | 112 | 1.5% |
| `aten::clamp_min` | 1 | 108 | 1.5% |
| `aten::fill_` | 16 | 14 | 0.2% |

## By category

| Category | us per forward | Share |
|---|---|---|
| attention (SDPA) | 878 | 12.0% |
| matmul / linear (addmm, mm, bmm) | 1341 | 18.3% |
| convolution | 1592 | 21.7% |
| normalization (LayerNorm, BatchNorm) | 970 | 13.2% |
| activation (GELU, ReLU) | 316 | 4.3% |
| interpolate / pooling | 578 | 7.9% |
| copy / layout (contiguous, cat, cast) | 1255 | 17.1% |
| elementwise (add, mul, div) | 396 | 5.4% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stage1 | 2.322 | 30.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.patch_embed | 0.176 | 2.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.attention | 0.765 | 9.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage1.attention.sr_conv+norm | 0.178 | 2.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.mix_ffn | 0.764 | 9.9% |
| stage2 | 1.131 | 14.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.patch_embed | 0.082 | 1.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.attention | 0.444 | 5.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage2.attention.sr_conv+norm | 0.087 | 1.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.mix_ffn | 0.427 | 5.5% |
| stage3 | 0.907 | 11.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.patch_embed | 0.059 | 0.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.attention | 0.366 | 4.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage3.attention.sr_conv+norm | 0.088 | 1.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.mix_ffn | 0.400 | 5.2% |
| stage4 | 0.587 | 7.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.patch_embed | 0.066 | 0.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.attention | 0.238 | 3.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.mix_ffn | 0.234 | 3.0% |
| decode_head | 2.637 | 34.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c1 | 0.193 | 2.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c2 | 0.062 | 0.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c3 | 0.033 | 0.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c4 | 0.017 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_fuse | 0.742 | 9.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.classifier | 0.073 | 0.9% |
| other (outside top-level parts: final upsample etc.) | 0.109 | 1.4% |
| **whole forward** | **7.694** | 100% |
