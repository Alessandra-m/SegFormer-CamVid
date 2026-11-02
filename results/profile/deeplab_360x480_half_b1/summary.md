# Profile: DeepLabV3-ResNet50, input 360x480, half, batch 1

| Metric | Value |
|---|---|
| latency without profiler, median | 20.65 ms |
| total GPU kernel time per forward | 20.57 ms |
| GPU busy = kernels / latency | 100% -> GPU-bound: kernels run almost back to back |
| CUDA kernel launches per forward | 397 |
| average kernel | 51.8 us |
| CPU time of aten ops per forward (with profiler overhead) | 4.89 ms |
| plus CUDA runtime CPU time (cudaLaunchKernel etc.) | 4.57 ms |

## Most expensive ops (self GPU time)

| Op | Calls per forward | us per forward | Share |
|---|---|---|---|
| `aten::cudnn_convolution` | 61 | 15007 | 73.0% |
| `aten::native_batch_norm` | 60 | 2287 | 11.1% |
| `aten::add_` | 17 | 1385 | 6.7% |
| `aten::clamp_min_` | 49 | 1381 | 6.7% |
| `aten::copy_` | 60 | 145 | 0.7% |
| `aten::upsample_bilinear2d` | 2 | 80 | 0.4% |
| `aten::clamp_min` | 7 | 79 | 0.4% |
| `aten::max_pool2d_with_indices` | 1 | 73 | 0.4% |
| `aten::cat` | 1 | 68 | 0.3% |
| `aten::mean` | 1 | 62 | 0.3% |

## By category

| Category | us per forward | Share |
|---|---|---|
| convolution | 15007 | 73.0% |
| normalization (LayerNorm, BatchNorm) | 2287 | 11.1% |
| activation (GELU, ReLU) | 1459 | 7.1% |
| interpolate / pooling | 152 | 0.7% |
| copy / layout (contiguous, cat, cast) | 213 | 1.0% |
| elementwise (add, mul, div) | 1385 | 6.7% |
| other | 62 | 0.3% |

## By model part (CUDA events in hooks)

| Part | ms per forward | Share of forward |
|---|---|---|
| stem (conv1+bn+relu+maxpool) | 0.390 | 1.9% |
| layer1 | 1.415 | 6.9% |
| layer2 | 1.477 | 7.2% |
| layer3 | 4.657 | 22.6% |
| layer4 | 6.776 | 32.8% |
| ASPP | 5.590 | 27.1% |
| head (conv3x3+bn+relu+conv1x1) | 0.281 | 1.4% |
| other (outside top-level parts: final upsample etc.) | 0.050 | 0.2% |
| **whole forward** | **20.635** | 100% |
