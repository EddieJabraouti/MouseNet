"""Conservative training-only descendants of raw TMT mouse recordings.

Synthetic labels are inherited from a real source participant. Evaluation
must hold out the source and every descendant together; no synthetic profile
is an independent participant or an independent diagnostic label.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .model import DEMOGRAPHICS, RAW_DIR, ROOT, SAMPLE_HZ, STRIDE_SAMPLES, WINDOW_SAMPLES, demographic_targets, load_checkpoint, parse_array, sha256
from .representation_variants import encoder_inputs, normalized_raw, resample
from .stable_mouse2vec import encode_batches


OUT = ROOT / "exp1/runs/synthetic_tmt"
REPLICAS = 2
SEED = 20260924


def demographics_by_id():
    with DEMOGRAPHICS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row["subject_id"]: np.array([int(float(row[field])) for field in ("sex", "age", "years_of_education")], dtype=np.int16) for row in rows}


def sample_demographics(base: np.ndarray, rng: np.random.Generator):
    sex, age, education = (int(v) for v in base)
    sampled = np.array((
        sex,
        np.clip(age + rng.integers(-2, 3), 55, 90),
        np.clip(education + rng.integers(-1, 2), 7, 20),
    ), dtype=np.float32)
    if sampled[0] != sex or abs(sampled[1] - age) > 2 or abs(sampled[2] - education) > 1:
        raise AssertionError("Synthetic demographic constraint failed")
    return sampled


def perturb_raw(t: np.ndarray, coords: np.ndarray, rng: np.random.Generator):
    duration_scale = rng.uniform(.98, 1.02)
    new_t = (t - t[0]) * duration_scale
    moving = np.r_[False, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 0]
    noise = rng.normal(0, .002, size=coords.shape)
    kernel = np.ones(9) / 9
    smooth = np.column_stack([np.convolve(noise[:, axis], kernel, mode="same") for axis in range(2)])
    new_coords = np.clip(coords + smooth * moving[:, None], 0, 1)
    if np.any(np.diff(new_t) <= 0) or not np.isfinite(new_coords).all():
        raise ValueError("Invalid synthetic trajectory")
    return new_t, new_coords, duration_scale


def generate(out_dir: Path = OUT, replicas: int = REPLICAS):
    if replicas < 1 or replicas > 4:
        raise ValueError("Replicas must be between 1 and 4")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "synthetic_hold_window_fft.npz"
    raw_target = out_dir / "synthetic_raw_trajectories.npz"
    if target.exists() or raw_target.exists():
        raise FileExistsError("Synthetic run already exists")
    rng = np.random.default_rng(SEED)
    labels = demographic_targets()
    demo_by_id = demographics_by_id()
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda path: int(path.name.split("_", 1)[0]))
    if {path.name.split("_", 1)[0] for path in files} != set(labels):
        raise ValueError("Raw/label ID mismatch")
    csv.field_size_limit(10_000_000)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    encoder = load_checkpoint()
    embeddings, records = [], []
    raw_t, raw_xy, raw_buttons, offsets = [], [], [], [0]
    raw_records = []
    duration_scales = []
    for number, path in enumerate(files, 1):
        sid = path.name.split("_", 1)[0]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = [row for row in csv.DictReader(handle) if row.get("blocks.thisIndex") == "1"]
        if len(source_rows) != 20:
            raise ValueError(f"Expected 20 trials for {sid}")
        for replica in range(replicas):
            synthetic_id = f"synthetic_{sid}_{replica + 1}"
            demo = sample_demographics(demo_by_id[sid], rng)
            windows, metadata = [], []
            for trial_order, row in enumerate(source_rows):
                part = row["trial_type"]
                if part != ("A" if trial_order % 2 == 0 else "B"):
                    raise ValueError("TMT trial order mismatch")
                t, xy = normalized_raw(row)
                buttons = np.column_stack([parse_array(row[f"trialMouse.{name}Button"], name) for name in ("left", "mid", "right")]).astype(np.int8)
                if len(buttons) != len(t):
                    raise ValueError("Button/time mismatch")
                synth_t, synth_xy, scale = perturb_raw(t, xy, rng)
                duration_scales.append(scale)
                raw_t.append(synth_t.astype(np.float32))
                raw_xy.append(synth_xy.astype(np.float32))
                raw_buttons.append(buttons)
                offsets.append(offsets[-1] + len(synth_t))
                raw_records.append((synthetic_id, sid, labels[sid], *demo, trial_order, part))
                _, sampled = resample(synth_t, synth_xy, "hold")
                presses = np.zeros(len(synth_t), dtype=bool)
                for axis in range(3):
                    button = buttons[:, axis]
                    presses |= (button > 0) & np.r_[True, button[:-1] == 0]
                for start in range(0, len(sampled) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
                    stop = start + WINDOW_SAMPLES
                    windows.append(sampled[start:stop])
                    clicks = int(np.sum((synth_t[presses] >= start / SAMPLE_HZ) & (synth_t[presses] < stop / SAMPLE_HZ)))
                    metadata.append((synthetic_id, sid, labels[sid], *demo, trial_order, part, start / SAMPLE_HZ, clicks))
            features = encoder_inputs(np.stack(windows), "window")
            embeddings.append(encode_batches(encoder, features))
            records.extend(metadata)
        if number % 10 == 0 or number == len(files):
            print(f"Synthetic TMT: encoded {number}/{len(files)} source participants", flush=True)
    z = np.concatenate(embeddings).astype(np.float32)
    columns = list(zip(*records))
    np.savez_compressed(
        target, embeddings=z,
        synthetic_subject_id=np.asarray(columns[0]), source_subject_id=np.asarray(columns[1]),
        group=np.asarray(columns[2], dtype=np.int8),
        demographics=np.asarray(columns[3:6], dtype=np.float32).T,
        trial_order=np.asarray(columns[6], dtype=np.int8), trial_type=np.asarray(columns[7]),
        window_start_s=np.asarray(columns[8], dtype=np.float32), click_count=np.asarray(columns[9], dtype=np.int16),
    )
    raw_columns = list(zip(*raw_records))
    np.savez_compressed(
        raw_target, timestamps_s=np.concatenate(raw_t), xy_normalized=np.concatenate(raw_xy),
        left_mid_right_button=np.concatenate(raw_buttons), trial_offsets=np.asarray(offsets, dtype=np.int64),
        synthetic_subject_id=np.asarray(raw_columns[0]), source_subject_id=np.asarray(raw_columns[1]),
        group=np.asarray(raw_columns[2], dtype=np.int8), demographics=np.asarray(raw_columns[3:6], dtype=np.float32).T,
        trial_order=np.asarray(raw_columns[6], dtype=np.int8), trial_type=np.asarray(raw_columns[7]),
    )
    if len(raw_records) != 74 * replicas * 20 or len(z) != len(records) or len(set(raw_columns[0])) != 74 * replicas:
        raise ValueError("Synthetic count audit failed")
    report = {
        "seed": SEED, "replicas_per_source": replicas, "n_source_participants": 74,
        "n_synthetic_profiles": 74 * replicas, "n_synthetic_trials": len(raw_records), "n_synthetic_windows": len(z),
        "age_bounds_years": [55, 90], "age_shift_max_years": 2,
        "education_bounds_years": [7, 20], "education_shift_max_years": 1,
        "sex_rule": "unchanged from source", "group_rule": "inherited from source",
        "trajectory_rule": "monotone global duration scale 0.98-1.02; smooth Gaussian x/y jitter sigma 0.002 only at moving samples; coordinates clipped to [0,1]; button states copied",
        "duration_scale_range_observed": [float(min(duration_scales)), float(max(duration_scales))],
        "encoder_inference": "frozen checkpoint with corrected token-axis modality embeddings",
        "raw_format": "Concatenated arrays; trial_offsets[i]:trial_offsets[i+1] yields synthetic raw trajectory i; x/y normalized to screen, timestamps relative seconds",
        "raw_sha256": sha256(raw_target), "embeddings_sha256": sha256(target),
        "class_counts_profiles": dict(Counter(labels[sid] for sid in labels for _ in range(replicas))),
        "warning": "Augmented descendants are correlated with their source and may not preserve every cognitive signal. Hold out real participants and all descendants together. Do not score synthetic profiles as independent test cases.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    args = parser.parse_args()
    generate(args.output, args.replicas)
