"""Reproduce the batch-axis modality indexing defect in cloned Mouse2Vec."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import RAW_DIR, ROOT, RUN_DIR, load_checkpoint, sha256
from .representation_variants import OUT, encoder_inputs, windows_for_participant
from .stable_mouse2vec import encode_batches


def audit():
    first = sorted(RAW_DIR.glob("*.csv"))[0]
    windows, _ = windows_for_participant(first, "hold")
    features = encoder_inputs(windows[:1], "window")
    repeated = np.repeat(features, 64, axis=0)
    model = load_checkpoint()
    model.eval()
    with torch.inference_mode():
        legacy = model.encoder(torch.from_numpy(repeated), model.opt).numpy()
    stable = encode_batches(model, repeated)
    with np.load(RUN_DIR / "window_embeddings.npz", allow_pickle=False) as data:
        old_z, old_ids, old_trials, old_starts = (data[field] for field in ("embeddings", "subject_id", "trial_order", "window_start_s"))
    stable_path = OUT / "hold_window_fft.npz"
    with np.load(stable_path, allow_pickle=False) as data:
        new_z, new_ids, new_trials, new_starts = (data[field] for field in ("embeddings", "subject_id", "trial_order", "window_start_s"))
    if not (np.array_equal(old_ids, new_ids) and np.array_equal(old_trials, new_trials) and np.allclose(old_starts, new_starts)):
        raise ValueError("Legacy/stable window metadata do not align")
    cosine = np.sum(old_z * new_z, axis=1) / (np.linalg.norm(old_z, axis=1) * np.linalg.norm(new_z, axis=1))
    report = {
        "source_code_location": "mouse2vec/Mouse2Vec/CRT.py TFR.forward: x[:m_token_idx], x[m_token_idx:p_token_idx], x[p_token_idx:] slice batch axis; intended operation slices token axis",
        "repeated_identical_input_count": len(repeated),
        "legacy_batch_position_max_abs_difference": float(np.max(np.abs(legacy - legacy[0]))),
        "corrected_batch_position_max_abs_difference": float(np.max(np.abs(stable - stable[0]))),
        "n_aligned_real_windows": len(old_z),
        "old_vs_corrected_cosine_similarity_quantiles": [float(v) for v in np.quantile(cosine, (0, .1, .5, .9, 1))],
        "legacy_embeddings_sha256": sha256(RUN_DIR / "window_embeddings.npz"),
        "corrected_embeddings_sha256": sha256(stable_path),
        "checkpoint_unchanged": True,
        "caution": "Corrected inference restores batch-position invariance but changes the representation relative to the released code; pretraining may itself have used the same defect. Downstream models must be refit and evaluated independently on corrected embeddings.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batching_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
