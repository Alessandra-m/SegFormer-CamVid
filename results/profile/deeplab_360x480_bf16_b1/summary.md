# Profile: DeepLabV3-ResNet50, input 360x480, bf16, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 22.02 ms |
| total GPU kernel time per forward | 21.92 ms |
| GPU busy = kernels / latency | 100% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 463 |
| average kernel | 47.3 us |
| CPU time of aten ops per forward (with profiler overhead) | 5.97 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 4.97 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 61 | 15045 | 68.6% |
| `aten::native_batch_norm` | 60 | 2244 | 10.2% |
| `aten::copy_` | 126 | 1442 | 6.6% |
| `aten::add_` | 17 | 1385 | 6.3% |
| `aten::clamp_min_` | 49 | 1378 | 6.3% |
| `aten::cat` | 1 | 120 | 0.5% |
| `aten::upsample_bilinear2d` | 4 | 90 | 0.4% |
| `aten::clamp_min` | 7 | 79 | 0.4% |
| `aten::max_pool2d_with_indices` | 1 | 73 | 0.3% |
| `aten::mean` | 1 | 62 | 0.3% |

## By category

| Category | us per forward | Share |
|---|---|---|
| convolution | 15045 | 68.6% |
| normalization (LayerNorm, BatchNorm) | 2244 | 10.2% |
| activation (GELU, ReLU) | 1457 | 6.6% |
| interpolate / pooling | 163 | 0.7% |
| copy / layout (contiguous, cat, cast) | 1562 | 7.1% |
| elementwise (add, mul, div) | 1385 | 6.3% |
| other | 62 | 0.3% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stem (conv1+bn+relu+maxpool) | 0.514 | 2.3% |
| layer1 | 1.498 | 6.6% |
| layer2 | 1.600 | 7.1% |
| layer3 | 4.913 | 21.8% |
| layer4 | 7.363 | 32.6% |
| ASPP | 6.272 | 27.8% |
| head (conv3x3+bn+relu+conv1x1) | 0.337 | 1.5% |
| other (outside top-level parts: final upsample etc.) | 0.066 | 0.3% |
| **whole forward** | **22.563** | 100% |
