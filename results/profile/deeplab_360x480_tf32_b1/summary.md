# Profile: DeepLabV3-ResNet50, input 360x480, tf32, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 38.53 ms |
| total GPU kernel time per forward | 40.46 ms |
| GPU busy = kernels / latency | 105% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 250 |
| average kernel | 161.8 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.47 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 9.72 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 61 | 29978 | 74.1% |
| `aten::cudnn_batch_norm` | 60 | 3873 | 9.6% |
| `aten::clamp_min_` | 49 | 3093 | 7.6% |
| `aten::add_` | 17 | 2801 | 6.9% |
| `aten::cat` | 1 | 222 | 0.5% |
| `aten::clamp_min` | 7 | 167 | 0.4% |
| `aten::max_pool2d_with_indices` | 1 | 117 | 0.3% |
| `aten::mean` | 1 | 115 | 0.3% |
| `aten::upsample_bilinear2d` | 2 | 90 | 0.2% |

## By category

| Category | us per forward | Share |
|---|---|---|
| convolution | 29978 | 74.1% |
| normalization (LayerNorm, BatchNorm) | 3873 | 9.6% |
| activation (GELU, ReLU) | 3261 | 8.1% |
| interpolate / pooling | 207 | 0.5% |
| copy / layout (contiguous, cat, cast) | 222 | 0.5% |
| elementwise (add, mul, div) | 2801 | 6.9% |
| other | 115 | 0.3% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stem (conv1+bn+relu+maxpool) | 0.512 | 1.3% |
| layer1 | 2.594 | 6.8% |
| layer2 | 2.477 | 6.5% |
| layer3 | 8.484 | 22.1% |
| layer4 | 12.950 | 33.8% |
| ASPP | 10.744 | 28.0% |
| head (conv3x3+bn+relu+conv1x1) | 0.524 | 1.4% |
| other (outside top-level parts: final upsample etc.) | 0.060 | 0.2% |
| **whole forward** | **38.345** | 100% |
