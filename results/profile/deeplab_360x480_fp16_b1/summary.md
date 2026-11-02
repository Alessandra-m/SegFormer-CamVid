# Profile: DeepLabV3-ResNet50, input 360x480, fp16, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 21.84 ms |
| total GPU kernel time per forward | 21.76 ms |
| GPU busy = kernels / latency | 100% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 346 |
| average kernel | 62.9 us |
| CPU time of aten ops per forward (with profiler overhead) | 5.51 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 4.90 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 61 | 14978 | 68.8% |
| `aten::cudnn_batch_norm` | 60 | 2250 | 10.3% |
| `aten::add_` | 17 | 1384 | 6.4% |
| `aten::clamp_min_` | 49 | 1374 | 6.3% |
| `aten::copy_` | 66 | 1351 | 6.2% |
| `aten::cat` | 1 | 120 | 0.6% |
| `aten::upsample_bilinear2d` | 4 | 91 | 0.4% |
| `aten::clamp_min` | 7 | 79 | 0.4% |
| `aten::max_pool2d_with_indices` | 1 | 73 | 0.3% |
| `aten::mean` | 1 | 63 | 0.3% |

## By category

| Category | us per forward | Share |
|---|---|---|
| convolution | 14978 | 68.8% |
| normalization (LayerNorm, BatchNorm) | 2250 | 10.3% |
| activation (GELU, ReLU) | 1453 | 6.7% |
| interpolate / pooling | 163 | 0.8% |
| copy / layout (contiguous, cat, cast) | 1471 | 6.8% |
| elementwise (add, mul, div) | 1384 | 6.4% |
| other | 63 | 0.3% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stem (conv1+bn+relu+maxpool) | 0.411 | 1.9% |
| layer1 | 1.393 | 6.4% |
| layer2 | 1.484 | 6.8% |
| layer3 | 4.826 | 22.1% |
| layer4 | 7.179 | 32.9% |
| ASPP | 6.183 | 28.3% |
| head (conv3x3+bn+relu+conv1x1) | 0.299 | 1.4% |
| other (outside top-level parts: final upsample etc.) | 0.064 | 0.3% |
| **whole forward** | **21.839** | 100% |
