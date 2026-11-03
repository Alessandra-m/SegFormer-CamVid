# SegFormer-B0 dependency groups (torch_pruning DepGraph)

Input 360x480, classifier ignored. Total groups: 31.

## Summary by type

| Type | Groups | Channels per group | Layers per group |
|---|---|---|---|
| decoder fuse | 1 | 256 | 3 |
| decoder projection | 4 | 256 | 2 |
| residual stream C | 4 | 32, 64, 160, 256 | 19, 20 |
| attention Q/K/V -> O | 8 | 32, 64, 160, 256 | 4 |
| key/value reduction | 6 | 32, 64, 160 | 4 |
| Mix-FFN hidden | 8 | 128, 256, 640, 1024 | 3 |

## All groups

| # | Type | Stage | Channels | Layers |
|---|---|---|---|---|
| 1 | decoder fuse | decoder | 256 | 3 |
| 2 | decoder projection | decoder | 256 | 2 |
| 3 | residual stream C | stage1 | 32 | 19 |
| 4 | attention Q/K/V -> O | stage1 | 32 | 4 |
| 5 | key/value reduction | stage1 | 32 | 4 |
| 6 | Mix-FFN hidden | stage1 | 128 | 3 |
| 7 | attention Q/K/V -> O | stage1 | 32 | 4 |
| 8 | key/value reduction | stage1 | 32 | 4 |
| 9 | Mix-FFN hidden | stage1 | 128 | 3 |
| 10 | decoder projection | decoder | 256 | 2 |
| 11 | residual stream C | stage2 | 64 | 19 |
| 12 | attention Q/K/V -> O | stage2 | 64 | 4 |
| 13 | key/value reduction | stage2 | 64 | 4 |
| 14 | Mix-FFN hidden | stage2 | 256 | 3 |
| 15 | attention Q/K/V -> O | stage2 | 64 | 4 |
| 16 | key/value reduction | stage2 | 64 | 4 |
| 17 | Mix-FFN hidden | stage2 | 256 | 3 |
| 18 | decoder projection | decoder | 256 | 2 |
| 19 | residual stream C | stage3 | 160 | 19 |
| 20 | attention Q/K/V -> O | stage3 | 160 | 4 |
| 21 | key/value reduction | stage3 | 160 | 4 |
| 22 | Mix-FFN hidden | stage3 | 640 | 3 |
| 23 | attention Q/K/V -> O | stage3 | 160 | 4 |
| 24 | key/value reduction | stage3 | 160 | 4 |
| 25 | Mix-FFN hidden | stage3 | 640 | 3 |
| 26 | decoder projection | decoder | 256 | 2 |
| 27 | residual stream C | stage4 | 256 | 20 |
| 28 | attention Q/K/V -> O | stage4 | 256 | 4 |
| 29 | Mix-FFN hidden | stage4 | 1024 | 3 |
| 30 | attention Q/K/V -> O | stage4 | 256 | 4 |
| 31 | Mix-FFN hidden | stage4 | 1024 | 3 |

## Coupled layers, one example per type

### decoder fuse (decoder, 256 channels)

- `dh.linear_fuse` (out channels)
- `dh.batch_norm` (out channels)
- `dh.classifier` (in channels)

### decoder projection (decoder, 256 channels)

- `dh.linear_projections.0.proj` (out channels)
- `dh.linear_fuse` (in channels)

### residual stream C (stage1, 32 channels)

- `s0.blocks.1.mlp.fc2` (out channels)
- `s0.layer_norm` (out channels)
- `s1.patch_embeddings.proj` (in channels)
- `dh.linear_projections.0.proj` (in channels)
- `s0.blocks.1.attention.o_proj` (out channels)
- `s0.blocks.1.layernorm_after` (out channels)
- `s0.blocks.1.mlp.fc1` (in channels)
- `s0.blocks.0.mlp.fc2` (out channels)
- `s0.blocks.1.layernorm_before` (out channels)
- `s0.blocks.1.attention.q_proj` (in channels)
- `s0.blocks.1.attention.sequence_reduction.sequence_reduction` (in channels)
- `s0.blocks.0.attention.o_proj` (out channels)
- `s0.patch_embeddings.layer_norm` (out channels)
- `s0.blocks.0.layernorm_after` (out channels)
- `s0.blocks.0.mlp.fc1` (in channels)
- `s0.blocks.0.layernorm_before` (out channels)
- `s0.blocks.0.attention.q_proj` (in channels)
- `s0.blocks.0.attention.sequence_reduction.sequence_reduction` (in channels)
- `s0.patch_embeddings.proj` (out channels)

### attention Q/K/V -> O (stage1, 32 channels)

- `s0.blocks.0.attention.v_proj` (out channels)
- `s0.blocks.0.attention.o_proj` (in channels)
- `s0.blocks.0.attention.k_proj` (out channels)
- `s0.blocks.0.attention.q_proj` (out channels)

### key/value reduction (stage1, 32 channels)

- `s0.blocks.0.attention.sequence_reduction.sequence_reduction` (out channels)
- `s0.blocks.0.attention.sequence_reduction.layer_norm` (out channels)
- `s0.blocks.0.attention.k_proj` (in channels)
- `s0.blocks.0.attention.v_proj` (in channels)

### Mix-FFN hidden (stage1, 128 channels)

- `s0.blocks.0.mlp.dwconv.dwconv` (out channels)
- `s0.blocks.0.mlp.fc2` (in channels)
- `s0.blocks.0.mlp.fc1` (out channels)

## Pitfall: attention heads

Pruning one group in `stage2.blocks.0.attention` (2 heads x head_dim 32 = 64 channels):

| Removed | q_proj weight after | Result |
|---|---|---|
| 1 channel | (63, 64) | forward fails: shape '[1, 2700, -1, 32]' is invalid for input of size 170100 |
| 32 channels, not aligned to a head | (32, 64) | forward OK, max abs logit diff = 2.32 |
| 32 channels = whole head 1 | (32, 64) | forward OK, max abs logit diff = 6.06 |
