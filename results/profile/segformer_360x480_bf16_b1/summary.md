# Profile: SegFormer-B0, input 360x480, bf16, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 7.62 ms |
| total GPU kernel time per forward | 5.74 ms |
| GPU busy = kernels / latency | 75% -> partly launch-bound: noticeable GPU idle gaps |
| CUDA kernel launches per forward | 430 |
| average kernel | 13.4 us |
| CPU time of aten ops per forward (with profiler overhead) | 6.94 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 2.02 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::copy_` | 235 | 1606 | 28.0% |
| `aten::native_layer_norm` | 30 | 864 | 15.1% |
| `aten::cudnn_convolution` | 12 | 625 | 10.9% |
| `aten::upsample_bilinear2d` | 10 | 579 | 10.1% |
| `aten::_conv_depthwise2d` | 8 | 448 | 7.8% |
| `aten::cat` | 1 | 434 | 7.6% |
| `aten::addmm` | 40 | 269 | 4.7% |
| `aten::mm` | 12 | 253 | 4.4% |
| `aten::_flash_attention_forward` | 8 | 161 | 2.8% |
| `aten::add_` | 23 | 147 | 2.6% |
| `aten::add` | 16 | 136 | 2.4% |
| `aten::gelu` | 8 | 101 | 1.8% |
| `aten::native_batch_norm` | 1 | 64 | 1.1% |
| `aten::clamp_min` | 1 | 55 | 1.0% |

## By category

| Category | us per forward | Share |
|---|---|---|
| attention (SDPA) | 161 | 2.8% |
| matmul / linear (addmm, mm, bmm) | 521 | 9.1% |
| convolution | 1073 | 18.7% |
| normalization (LayerNorm, BatchNorm) | 929 | 16.2% |
| activation (GELU, ReLU) | 156 | 2.7% |
| interpolate / pooling | 579 | 10.1% |
| copy / layout (contiguous, cat, cast) | 2040 | 35.5% |
| elementwise (add, mul, div) | 283 | 4.9% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stage1 | 1.651 | 17.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.patch_embed | 0.197 | 2.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.attention | 0.330 | 3.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage1.attention.sr_conv+norm | 0.167 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.mix_ffn | 0.542 | 5.8% |
| stage2 | 1.185 | 12.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.patch_embed | 0.082 | 0.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.attention | 0.457 | 4.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage2.attention.sr_conv+norm | 0.156 | 1.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.mix_ffn | 0.395 | 4.2% |
| stage3 | 2.080 | 22.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.patch_embed | 0.169 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.attention | 1.062 | 11.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage3.attention.sr_conv+norm | 0.364 | 3.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.mix_ffn | 0.487 | 5.2% |
| stage4 | 1.637 | 17.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.patch_embed | 0.176 | 1.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.attention | 0.614 | 6.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.mix_ffn | 0.490 | 5.3% |
| decode_head | 2.495 | 26.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c1 | 0.182 | 2.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c2 | 0.045 | 0.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c3 | 0.029 | 0.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c4 | 0.020 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_fuse | 0.649 | 7.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.classifier | 0.044 | 0.5% |
| other (outside top-level parts: final upsample etc.) | 0.267 | 2.9% |
| **whole forward** | **9.314** | 100% |
