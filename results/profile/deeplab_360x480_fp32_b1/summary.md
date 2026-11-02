# Profile: DeepLabV3-ResNet50, input 360x480, fp32, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 38.02 ms |
| total GPU kernel time per forward | 38.05 ms |
| GPU busy = kernels / latency | 100% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 247 |
| average kernel | 154.0 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.23 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 9.22 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 61 | 28018 | 73.6% |
| `aten::cudnn_batch_norm` | 60 | 3786 | 10.0% |
| `aten::clamp_min_` | 49 | 2890 | 7.6% |
| `aten::add_` | 17 | 2749 | 7.2% |
| `aten::clamp_min` | 7 | 169 | 0.4% |
| `aten::cat` | 1 | 136 | 0.4% |
| `aten::mean` | 1 | 115 | 0.3% |
| `aten::max_pool2d_with_indices` | 1 | 96 | 0.3% |
| `aten::upsample_bilinear2d` | 2 | 91 | 0.2% |

## By category

| Category | us per forward | Share |
|---|---|---|
| convolution | 28018 | 73.6% |
| normalization (LayerNorm, BatchNorm) | 3786 | 10.0% |
| activation (GELU, ReLU) | 3059 | 8.0% |
| interpolate / pooling | 187 | 0.5% |
| copy / layout (contiguous, cat, cast) | 136 | 0.4% |
| elementwise (add, mul, div) | 2749 | 7.2% |
| other | 115 | 0.3% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stem (conv1+bn+relu+maxpool) | 0.517 | 1.4% |
| layer1 | 2.561 | 6.7% |
| layer2 | 2.442 | 6.4% |
| layer3 | 8.468 | 22.2% |
| layer4 | 12.827 | 33.7% |
| ASPP | 10.679 | 28.0% |
| head (conv3x3+bn+relu+conv1x1) | 0.540 | 1.4% |
| other (outside top-level parts: final upsample etc.) | 0.061 | 0.2% |
| **whole forward** | **38.094** | 100% |
