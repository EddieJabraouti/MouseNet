"""Alternative frozen Mouse2Vec inputs with participant-local FFT scaling.

The official helper scales spectra across a batch of windows. The historical
adapter scaled each window separately. Here the scope is one participant's
completed 20-trial session, so no other participant affects an embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .model import (
    RAW_DIR, ROOT, SAMPLE_HZ, STRIDE_SAMPLES, WINDOW_SAMPLES,
    demographic_targets, load_checkpoint, parse_array, sha256,
)
from .stable_mouse2vec import encode_batches


OUT = ROOT / "exp1/runs/representation_variants"


def normalized_raw(row):
    t = parse_array(row["trialMouse.time"], "time")
    x = parse_array(row["trialMouse.x"], "x")
    y = parse_array(row["trialMouse.y"], "y")
    if len(t) < 2 or len(t) != len(x) or len(t) != len(y) or np.any(np.diff(t) <= 0):
        raise ValueError("Bad raw mouse series")
    coords = np.column_stack(((x + 8 / 9) / (16 / 9), .5 - y))
    if np.any(coords < -1e-4) or np.any(coords > 1 + 1e-4):
        raise ValueError("Coordinate outside TMT screen")
    return t, np.clip(coords, 0, 1)


def resample(t, coords, method: str):
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * SAMPLE_HZ)) + 1) / SAMPLE_HZ
    if method == "hold":
        index = np.maximum(np.searchsorted(t, grid, side="right") - 1, 0)
        sampled = coords[index]
    elif method == "linear":
        sampled = np.column_stack((np.interp(grid, t, coords[:, 0]), np.interp(grid, t, coords[:, 1])))
    else:
        raise ValueError(method)
    return grid, sampled


def windows_for_participant(path: Path, method: str):
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    windows, metadata = [], []
    trial_order = 0
    for row_number, row in enumerate(rows):
        if row.get("blocks.thisIndex") != "1":
            continue
        part = row["trial_type"]
        if part != ("A" if trial_order % 2 == 0 else "B"):
            raise ValueError("Unexpected TMT order")
        t, coords = normalized_raw(row)
        _, sampled = resample(t, coords, method)
        buttons = [parse_array(row[f"trialMouse.{name}Button"], name) for name in ("left", "mid", "right")]
        presses = np.zeros(len(t), dtype=bool)
        for button in buttons:
            if len(button) != len(t):
                raise ValueError("Button/time mismatch")
            presses |= (button > 0) & np.r_[True, button[:-1] == 0]
        for start in range(0, len(sampled) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            stop = start + WINDOW_SAMPLES
            windows.append(sampled[start:stop])
            click_count = int(np.sum((t[presses] >= t[0] + start / SAMPLE_HZ) & (t[presses] < t[0] + stop / SAMPLE_HZ)))
            metadata.append((row_number, trial_order, part, start / SAMPLE_HZ, stop / SAMPLE_HZ, click_count))
        trial_order += 1
    if trial_order != 20:
        raise ValueError("Missing experimental trials")
    return np.stack(windows), metadata


def encoder_inputs(windows: np.ndarray, scope: str):
    if windows.ndim != 3 or windows.shape[1:] != (100, 2):
        raise ValueError("Bad Mouse2Vec window array")
    freq = np.fft.fft(windows, axis=1)[:, :50, :]
    outputs = []
    for values in (np.abs(freq), np.angle(freq)):
        if scope == "participant":
            low = values.min(axis=(0, 1), keepdims=True)
            high = values.max(axis=(0, 1), keepdims=True)
        elif scope == "window":
            low = values.min(axis=1, keepdims=True)
            high = values.max(axis=1, keepdims=True)
        else:
            raise ValueError(scope)
        outputs.append(np.divide(values - low, high - low, out=np.zeros_like(values), where=high > low))
    features = np.concatenate((windows, *outputs), axis=1).transpose(0, 2, 1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite encoder features")
    return features


def encode_variant(method: str, scope: str, out_dir: Path = OUT):
    name = f"{method}_{scope}_fft"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{name}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    targets = demographic_targets()
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda path: int(path.name.split("_", 1)[0]))
    if {path.name.split("_", 1)[0] for path in files} != set(targets):
        raise ValueError("Raw/demographic ID mismatch")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_checkpoint()
    embeddings, records = [], []
    for number, path in enumerate(files, 1):
        sid = path.name.split("_", 1)[0]
        windows, metadata = windows_for_participant(path, method)
        features = encoder_inputs(windows, scope)
        embeddings.append(encode_batches(model, features))
        records.extend((sid, path.name, *entry) for entry in metadata)
        if number % 10 == 0 or number == len(files):
            print(f"{name}: encoded {number}/{len(files)} participants", flush=True)
    z = np.concatenate(embeddings).astype(np.float32)
    if len(z) != len(records) or len(z) != 26655:
        raise ValueError("Unexpected variant window count")
    fields = list(zip(*records))
    np.savez_compressed(
        destination, embeddings=z,
        subject_id=np.asarray(fields[0]), source_file=np.asarray(fields[1]),
        row_number=np.asarray(fields[2], dtype=np.int16),
        trial_order=np.asarray(fields[3], dtype=np.int8), trial_type=np.asarray(fields[4]),
        window_start_s=np.asarray(fields[5], dtype=np.float32),
        window_end_s=np.asarray(fields[6], dtype=np.float32), click_count=np.asarray(fields[7], dtype=np.int16),
    )
    report = {
        "name": name, "coordinate_resampling": method, "fft_scaling_scope": scope,
        "encoder_inference": "stable modality embeddings applied along token axis; official checkpoint weights unchanged",
        "fft_scaling_detail": "min-max separately per x/y axis across one completed participant session" if scope == "participant" else "min-max separately per x/y axis within each 5-second window",
        "mouse2vec_checkpoint_sha256": sha256(ROOT / "mouse2vec/Mouse2Vec/pretrained_model.pkl"),
        "n_participants": 74, "n_windows": len(z), "embedding_shape": list(z.shape),
        "class_counts": dict(Counter(targets.values())),
        "warning": "Participant-local scaling assumes all 20 TMT trials are available at prediction time. No scaling uses other participants or labels." if scope == "participant" else "Per-window scaling uses only the window being encoded.",
    }
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resampling", choices=("hold", "linear"), required=True)
    parser.add_argument("--fft-scope", choices=("window", "participant"), required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    encode_variant(args.resampling, args.fft_scope, args.output)
