| Model | Input | Variant | mIoU test | Δ mIoU | Latency b=1, ms (median / p90) | Speedup | FPS, b=8 | Memory b=1, MB (weights + activations) | Params, M | GMACs |
|---|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B0 | 512x683 | fp32 | 0.7127 | +0.0000 | 16.31 / 16.41 | ×1.00 | 78.1 | 215 (14 + 201) | 3.72 | 10.9 |
| SegFormer-B0 | 512x683 | fp16 | 0.7127 | -0.0000 | 11.45 / 11.56 | ×1.42 | 118.6 | 247 (14 + 233) | 3.72 | 10.9 |
| SegFormer-B0 | 512x683 | half | 0.7127 | -0.0000 | 8.13 / 8.21 | ×2.01 | 163.3 | 106 (7 + 99) | 3.72 | 10.9 |
| SegFormer-B0 | 512x683 | fp16+compile | 0.7127 | -0.0000 | 6.31 / 6.44 | ×2.58 | 133.4 | — | 3.72 | 10.9 |
