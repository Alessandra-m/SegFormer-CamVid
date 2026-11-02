# Profile: SegFormer-B0, input 360x480, half, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 5.76 ms |
| total GPU kernel time per forward | 4.02 ms |
| GPU busy = kernels / latency | 70% -> partly launch-bound: noticeable GPU idle gaps |
| CUDA kernel launches per forward | 232 |
| average kernel | 17.3 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.57 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 1.05 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 20 | 773 | 19.2% |
| `aten::native_layer_norm` | 30 | 729 | 18.1% |
| `aten::upsample_bilinear2d` | 5 | 564 | 14.0% |
| `aten::copy_` | 30 | 541 | 13.4% |
| `aten::addmm` | 40 | 285 | 7.1% |
| `aten::add_` | 31 | 261 | 6.5% |
| `aten::mm` | 12 | 219 | 5.4% |
| `aten::cat` | 1 | 216 | 5.4% |
| `aten::_flash_attention_forward` | 8 | 159 | 4.0% |
| `aten::gelu` | 8 | 98 | 2.4% |
| `aten::native_batch_norm` | 1 | 64 | 1.6% |
| `aten::add` | 16 | 60 | 1.5% |
| `aten::clamp_min` | 1 | 54 | 1.4% |

## By category

| Category | us per forward | Share |
|---|---|---|
| attention (SDPA) | 159 | 4.0% |
| matmul / linear (addmm, mm, bmm) | 504 | 12.5% |
| convolution | 773 | 19.2% |
| normalization (LayerNorm, BatchNorm) | 794 | 19.7% |
| activation (GELU, ReLU) | 152 | 3.8% |
| interpolate / pooling | 564 | 14.0% |
| copy / layout (contiguous, cat, cast) | 757 | 18.8% |
| elementwise (add, mul, div) | 321 | 8.0% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stage1 | 1.344 | 18.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.patch_embed | 0.147 | 2.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.attention | 0.268 | 3.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage1.attention.sr_conv+norm | 0.147 | 2.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage1.mix_ffn | 0.440 | 5.9% |
| stage2 | 1.264 | 16.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.patch_embed | 0.063 | 0.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.attention | 0.525 | 7.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage2.attention.sr_conv+norm | 0.192 | 2.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage2.mix_ffn | 0.382 | 5.1% |
| stage3 | 1.671 | 22.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.patch_embed | 0.131 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.attention | 0.791 | 10.6% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stage3.attention.sr_conv+norm | 0.293 | 3.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage3.mix_ffn | 0.398 | 5.3% |
| stage4 | 1.296 | 17.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.patch_embed | 0.133 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.attention | 0.397 | 5.3% |
| &nbsp;&nbsp;&nbsp;&nbsp;stage4.mix_ffn | 0.416 | 5.6% |
| decode_head | 1.649 | 22.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c1 | 0.138 | 1.8% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c2 | 0.034 | 0.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c3 | 0.018 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_c4 | 0.013 | 0.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.linear_fuse | 0.328 | 4.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;decode_head.classifier | 0.040 | 0.5% |
| other (outside top-level parts: final upsample etc.) | 0.253 | 3.4% |
| **whole forward** | **7.478** | 100% |
