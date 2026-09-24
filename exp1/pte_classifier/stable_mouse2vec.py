"""Batch-position-invariant inference for the frozen Mouse2Vec checkpoint.

The cloned TFR.forward adds modality embeddings with x[:index], which slices
the batch axis. The intended slice is x[:, :index], along the token axis.
This adapter fixes that inference operation without modifying third-party code
or weights. It must not be mixed with the older saved embedding artifact.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import repeat
from einops.layers.torch import Rearrange


def stable_encoder_forward(model, features: torch.Tensor) -> torch.Tensor:
    encoder = model.encoder
    batch, channels, steps = features.shape
    if channels != 2 or steps != 200:
        raise ValueError("Expected Mouse2Vec [batch, 2, 200] features")
    t, m, p = features[:, :, :steps // 2], features[:, :, steps // 2: steps * 3 // 4], features[:, :, -steps // 4:]
    if any(signal.shape[-1] % model.opt.patch_len for signal in (t, m, p)):
        raise ValueError("Window cannot be patched")
    t, m, p = encoder.to_patch(t), encoder.to_patch(m), encoder.to_patch(p)
    pool = torch.nn.AdaptiveAvgPool1d(1)
    restore = Rearrange('(b n) c 1 -> b n c', b=batch)
    cls = repeat(encoder.cls_token, '() n d -> b n d', b=batch)
    x = torch.cat((
        cls[:, 0:1, :], restore(pool(encoder.cnn1(t))),
        cls[:, 1:2, :], restore(pool(encoder.cnn2(m))),
        cls[:, 2:3, :], restore(pool(encoder.cnn3(p))),
    ), dim=1)
    n_patches = x.shape[1] - 3
    m_index = n_patches // 2 + 1
    p_index = n_patches * 3 // 4 + 2
    x[:, :m_index, :] += encoder.modal_embedding[:1]
    x[:, m_index:p_index, :] += encoder.modal_embedding[1:2]
    x[:, p_index:, :] += encoder.modal_embedding[2:]
    x += encoder.pos_embedding[:, :x.shape[1]]
    x = encoder.dropout(x)
    x = encoder.transformer(x)
    return (x[:, 0] + x[:, m_index] + x[:, p_index]) / 3


def encode_batches(model, features: np.ndarray, batch_size: int = 64) -> np.ndarray:
    if features.ndim != 3 or features.shape[1:] != (2, 200):
        raise ValueError("Invalid Mouse2Vec feature array")
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            z = stable_encoder_forward(model, torch.from_numpy(features[start:start + batch_size])).cpu().numpy()
            if z.shape != (min(batch_size, len(features) - start), 128) or not np.isfinite(z).all():
                raise ValueError("Invalid stable embedding")
            vectors.append(z)
    return np.concatenate(vectors).astype(np.float32)
