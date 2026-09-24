"""Frozen Mouse2Vec extraction and historical averaged-feature evaluation.

The encoder is never fitted on TMT. Current PLS/SVM evaluation is in
window_pls_svm.py; this module preserves the earlier averaged-feature run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/nm9xy-osfstorage-raw_data-archive"
DEMOGRAPHICS = ROOT / "data/nm9xy-osfstorage-processed_data-archive/demographic_df.csv"
MOUSE2VEC_DIR = ROOT / "mouse2vec/Mouse2Vec"
CHECKPOINT = MOUSE2VEC_DIR / "pretrained_model.pkl"
RUN_DIR = ROOT / "exp1/runs/frozen_mouse2vec"
RBF_RUN_DIR = ROOT / "exp1/runs/rbf_mouse2vec_demographics"
SAMPLE_HZ = 20
WINDOW_SECONDS = 5
STRIDE_SECONDS = 1
WINDOW_SAMPLES = SAMPLE_HZ * WINDOW_SECONDS
STRIDE_SAMPLES = SAMPLE_HZ * STRIDE_SECONDS
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint():
    import torch

    # Full-model pickle from the official Mouse2Vec repository. Import its
    # classes before loading; only load this trusted, local checkpoint.
    sys.path.insert(0, str(MOUSE2VEC_DIR))
    import CRT  # noqa: F401

    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    # PyTorch 1.10 serialized GELU without this attribute; PyTorch 2.8 needs it.
    for layer in model.modules():
        if isinstance(layer, torch.nn.GELU) and not hasattr(layer, "approximate"):
            layer.approximate = "none"
    model.eval()
    if model.opt.seq_len != 200 or model.opt.dim != 128:
        raise ValueError("Unexpected Mouse2Vec checkpoint dimensions")
    return model


def parse_array(value: str, name: str) -> np.ndarray:
    array = np.asarray(ast.literal_eval(value), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name} array")
    return array


def mouse2vec_input(coords: np.ndarray) -> np.ndarray:
    """Create the checkpoint's [x/y, time/magnitude/phase] input per window.

    The released DFT helper min-max scales spectra over its input batch. We
    pass one window at a time to keep this transform independent of other
    participants, with a defined zero result for constant spectra.
    """
    if coords.shape != (WINDOW_SAMPLES, 2):
        raise ValueError("Window must contain 100 x/y samples")
    freq = np.fft.fft(coords, axis=0)[: WINDOW_SAMPLES // 2]

    def scale(values: np.ndarray) -> np.ndarray:
        low = values.min(axis=0)
        span = values.max(axis=0) - low
        return np.divide(values - low, span, out=np.zeros_like(values), where=span > 0)

    magnitude = scale(np.abs(freq))
    phase = scale(np.angle(freq))
    out = np.concatenate((coords, magnitude, phase), axis=0).T.astype(np.float32)
    if out.shape != (2, 200) or not np.isfinite(out).all():
        raise ValueError("Invalid Mouse2Vec input")
    return out


def trial_windows(row: dict[str, str]):
    x = parse_array(row["trialMouse.x"], "x")
    y = parse_array(row["trialMouse.y"], "y")
    t = parse_array(row["trialMouse.time"], "time")
    buttons = [parse_array(row[f"trialMouse.{name}Button"], name) for name in ("left", "mid", "right")]
    if not all(len(a) == len(t) for a in (x, y, *buttons)) or len(t) < 2:
        raise ValueError("Trial arrays have inconsistent lengths")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trial timestamps are not strictly increasing")

    # PsychoPy 'height' coordinates: observed limits are x +/-8/9, y +/-1/2.
    # Mouse2Vec uses on-screen coordinates in [0, 1], with y downwards.
    coords = np.column_stack(((x + 8 / 9) / (16 / 9), 0.5 - y))
    if np.any(coords < -1e-4) or np.any(coords > 1 + 1e-4):
        raise ValueError("Coordinate exceeds inferred 16:9 TMT window bounds")
    coords = np.clip(coords, 0, 1)

    sample_count = int(np.floor((t[-1] - t[0]) * SAMPLE_HZ)) + 1
    grid = t[0] + np.arange(sample_count) / SAMPLE_HZ
    source_idx = np.maximum(np.searchsorted(t, grid, side="right") - 1, 0)
    sampled = coords[source_idx]

    # The encoder consumes trajectories, while the pretraining click task used
    # button events. Keep click counts in metadata for extraction auditing.
    press = np.zeros(len(t), dtype=bool)
    for button in buttons:
        press |= (button > 0) & np.r_[True, button[:-1] == 0]
    click_grid = np.zeros(sample_count, dtype=np.int16)
    click_idx = np.clip(np.floor((t[press] - t[0]) * SAMPLE_HZ).astype(int), 0, sample_count - 1)
    np.add.at(click_grid, click_idx, 1)

    for start in range(0, sample_count - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        stop = start + WINDOW_SAMPLES
        yield (
            mouse2vec_input(sampled[start:stop]),
            float(start / SAMPLE_HZ),
            float(stop / SAMPLE_HZ),
            int(click_grid[start:stop].sum()),
        )


def demographic_targets() -> dict[str, int]:
    with DEMOGRAPHICS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    targets = {row["subject_id"]: int(row["group"]) for row in rows}
    if len(targets) != len(rows) or set(targets.values()) != {0, 1}:
        raise ValueError("Invalid demographic subject IDs or targets")
    return targets


def encode(out_dir: Path, batch_size: int) -> None:
    import torch

    if (out_dir / "window_embeddings.npz").exists():
        raise FileExistsError(f"Extraction already exists in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: int(p.name.split("_", 1)[0]))
    targets = demographic_targets()
    file_ids = {path.name.split("_", 1)[0] for path in files}
    if file_ids != set(targets):
        raise ValueError("Raw filename IDs do not match processed participant IDs")
    if len(files) != 74:
        raise ValueError(f"Expected 74 TMT files; found {len(files)}")

    model = load_checkpoint()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pending: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    records: list[tuple] = []
    mismatches = []
    trial_count = Counter()

    def flush() -> None:
        if not pending:
            return
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(pending))
            z = model.encoder(x, model.opt).cpu().numpy()
        if z.shape != (len(pending), 128) or not np.isfinite(z).all():
            raise ValueError("Invalid encoder output")
        vectors.extend(z)
        pending.clear()

    csv.field_size_limit(10_000_000)
    for file_number, path in enumerate(files, 1):
        subject_id = path.name.split("_", 1)[0]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        raw_ids = {r["participant"] for r in rows if r.get("participant")}
        if len(raw_ids) != 1:
            raise ValueError(f"Ambiguous internal participant in {path.name}")
        raw_id = next(iter(raw_ids))
        if raw_id != subject_id:
            mismatches.append({"file_id": subject_id, "internal_id": raw_id, "file": path.name})
        trial_order = 0
        for row_number, row in enumerate(rows):
            if row.get("blocks.thisIndex") != "1":
                continue  # skip both practice trials and session metadata
            part = row.get("trial_type")
            if part not in ("A", "B"):
                raise ValueError(f"Unexpected trial type in {path.name}")
            trial_count[(subject_id, part)] += 1
            count_this_trial = 0
            for features, start, stop, clicks in trial_windows(row):
                pending.append(features)
                records.append((subject_id, raw_id, path.name, row_number, trial_order, part, start, stop, clicks))
                count_this_trial += 1
                if len(pending) >= batch_size:
                    flush()
            if count_this_trial == 0:
                raise ValueError(f"No complete Mouse2Vec window in {path.name}, row {row_number}")
            trial_order += 1
        if trial_order != 20:
            raise ValueError(f"Expected 20 experimental trials in {path.name}")
        if file_number % 10 == 0 or file_number == len(files):
            print(f"Encoded {file_number}/{len(files)} participants", flush=True)
    flush()
    if len(vectors) != len(records):
        raise ValueError("Embedding/metadata count mismatch")
    if any(trial_count[(sid, part)] != 10 for sid in file_ids for part in ("A", "B")):
        raise ValueError("Incomplete A/B trial set")

    z = np.stack(vectors).astype(np.float32)
    fields = list(zip(*records))
    np.savez_compressed(
        out_dir / "window_embeddings.npz",
        embeddings=z,
        subject_id=np.asarray(fields[0]),
        raw_participant=np.asarray(fields[1]),
        source_file=np.asarray(fields[2]),
        row_number=np.asarray(fields[3], dtype=np.int16),
        trial_order=np.asarray(fields[4], dtype=np.int8),
        trial_type=np.asarray(fields[5]),
        window_start_s=np.asarray(fields[6], dtype=np.float32),
        window_end_s=np.asarray(fields[7], dtype=np.float32),
        click_count=np.asarray(fields[8], dtype=np.int16),
    )

    ids = sorted(file_ids, key=int)
    y = np.asarray([targets[sid] for sid in ids], dtype=np.int8)

    manifest = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "encoder_trainable": False,
        "sample_hz": SAMPLE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "coordinate_transform": "PsychoPy height 16:9 to Mouse2Vec [0,1], y down",
        "frequency_scaling": "per-window, per-axis min-max of FFT magnitude and phase",
        "participants": len(ids),
        "class_counts": dict(Counter(int(v) for v in y)),
        "experimental_trials": int(sum(trial_count.values())),
        "windows": int(len(z)),
        "embedding_dim": int(z.shape[1]),
        "internal_id_mismatches": mismatches,
    }
    (out_dir / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("participants", "experimental_trials", "windows", "embedding_dim")}, indent=2))


def demographic_features(ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    with DEMOGRAPHICS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["subject_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("Demographic IDs do not match participant features")
    if any(int(by_id[sid]["group"]) != label for sid, label in zip(ids, y)):
        raise ValueError("Demographic targets do not match participant features")
    demographics = np.asarray(
        [[float(by_id[sid][field]) for field in ("sex", "age", "years_of_education")] for sid in ids],
        dtype=np.float64,
    )
    if demographics.shape != (74, 3) or not np.isfinite(demographics).all():
        raise ValueError("Invalid demographic feature matrix")
    if not set(demographics[:, 0]).issubset({0.0, 1.0}):
        raise ValueError("Unexpected sex coding")
    return demographics


def evaluate_averaged_baseline(out_dir: Path, features_path: Path) -> None:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(features_path, allow_pickle=False) as data:
        mouse, y, ids = data["X"], data["y"].astype(int), data["subject_id"]
    if mouse.shape != (74, 256) or len(set(ids)) != 74 or Counter(y) != Counter({0: 33, 1: 41}):
        raise ValueError("Participant matrix fails cohort audit")
    if not np.isfinite(mouse).all():
        raise ValueError("Non-finite classifier features")
    X = np.column_stack((mouse, demographic_features(ids, y)))

    # On this macOS NumPy/Accelerate build, finite small matrix products can
    # emit floating-point warnings while agreeing with explicit dot sums to
    # machine precision. Keep the finiteness checks above and below.
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning, module="sklearn.utils.extmath"
    )

    def search(inner_folds: int = 10) -> GridSearchCV:
        preprocessing = ColumnTransformer(
            [
                ("mouse", StandardScaler(), slice(0, 256)),
                ("demographics", StandardScaler(), slice(256, 259)),
            ],
            transformer_weights={"mouse": 1 / np.sqrt(256), "demographics": 1 / np.sqrt(3)},
        )
        pipeline = make_pipeline(
            preprocessing,
            SVC(kernel="rbf", class_weight="balanced"),
        )
        return GridSearchCV(
            pipeline,
            {"svc__C": [0.1, 1.0, 10.0, 100.0], "svc__gamma": [0.1, 1.0, 10.0]},
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED),
            n_jobs=1,
            refit=True,
        )

    probabilities = np.full(len(y), np.nan)
    chosen_c = np.full(len(y), np.nan)
    chosen_gamma = np.full(len(y), np.nan)
    inner_auc = np.full(len(y), np.nan)
    for fold, (train, test) in enumerate(LeaveOneOut().split(X, y), 1):
        if len(test) != 1 or set(ids[train]) & set(ids[test]):
            raise AssertionError("Participant overlap in outer fold")
        model = search().fit(X[train], y[train])
        calibrated = CalibratedClassifierCV(
            estimator=model.best_estimator_,
            method="sigmoid",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        ).fit(X[train], y[train])
        probabilities[test] = calibrated.predict_proba(X[test])[:, 1]
        chosen_c[test] = model.best_params_["svc__C"]
        chosen_gamma[test] = model.best_params_["svc__gamma"]
        inner_auc[test] = model.best_score_
        if fold % 10 == 0 or fold == len(y):
            print(f"Evaluated {fold}/{len(y)} held-out participants", flush=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("Missing out-of-fold prediction")
    predictions = (probabilities >= 0.5).astype(int)
    auc = float(roc_auc_score(y, probabilities))
    balanced = float(balanced_accuracy_score(y, predictions))
    sensitivity = float(predictions[y == 1].mean())
    specificity = float(1 - predictions[y == 0].mean())

    rng = np.random.default_rng(SEED)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot_auc, boot_balanced = [], []
    for _ in range(2000):
        sample = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        boot_auc.append(roc_auc_score(y[sample], probabilities[sample]))
        boot_balanced.append(balanced_accuracy_score(y[sample], predictions[sample]))

    # Deployment artifact: fit on all available participants. Its measured
    # performance remains the nested, out-of-fold estimate above.
    final = search().fit(X, y)
    final_calibrated = CalibratedClassifierCV(
        estimator=final.best_estimator_,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    ).fit(X, y)
    joblib.dump(final_calibrated, out_dir / "final_classifier.joblib")
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "group", "probability_group_1", "predicted_group", "chosen_C", "chosen_gamma", "inner_auc"))
        writer.writerows(zip(ids, y, probabilities, predictions, chosen_c, chosen_gamma, inner_auc))

    # LOOCV refits on slightly different class prevalences for positive and
    # negative held-out cases. Repeated stratified folds check whether that
    # shift distorts the pooled LOOCV probability ranking.
    fivefold_results = []
    with (out_dir / "fivefold_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("repeat_seed", "subject_id", "group", "probability_group_1", "predicted_group"))
        for repeat_seed in range(SEED, SEED + 5):
            fivefold_probabilities = np.full(len(y), np.nan)
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
            for train, test in outer.split(X, y):
                selected = search(inner_folds=5).fit(X[train], y[train])
                calibrated = CalibratedClassifierCV(
                    estimator=selected.best_estimator_,
                    method="sigmoid",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                ).fit(X[train], y[train])
                fivefold_probabilities[test] = calibrated.predict_proba(X[test])[:, 1]
            if not np.isfinite(fivefold_probabilities).all():
                raise ValueError("Missing repeated five-fold prediction")
            fivefold_predictions = (fivefold_probabilities >= 0.5).astype(int)
            fivefold_results.append(
                {
                    "seed": repeat_seed,
                    "auroc": float(roc_auc_score(y, fivefold_probabilities)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, fivefold_predictions)),
                }
            )
            writer.writerows(zip([repeat_seed] * len(y), ids, y, fivefold_probabilities, fivefold_predictions))

    report = {
        "protocol": "outer leave-one-participant-out; inner stratified 10-fold AUROC tuning",
        "model": "class-balanced RBF SVM with fold-local block standardization and sigmoid calibration",
        "feature_definition": "256 Mouse2Vec A/B summary features plus sex, age, years of education; each standardized block weighted by inverse square root of its feature count",
        "selection": "C in [0.1, 1, 10, 100] and gamma in [0.1, 1, 10] selected on inner folds only",
        "calibration": "5-fold stratified sigmoid calibration fitted on each outer training set only, after inner C/gamma selection",
        "decision_threshold": 0.5,
        "mouse_features_sha256": sha256(features_path),
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "n_participants": int(len(y)),
        "class_counts": dict(Counter(int(v) for v in y)),
        "auroc": auc,
        "auroc_bootstrap_95pct": [float(v) for v in np.quantile(boot_auc, (0.025, 0.975))],
        "balanced_accuracy": balanced,
        "balanced_accuracy_bootstrap_95pct": [float(v) for v in np.quantile(boot_balanced, (0.025, 0.975))],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "final_model_C": float(final.best_params_["svc__C"]),
        "final_model_gamma": float(final.best_params_["svc__gamma"]),
        "bootstrap_note": "Stratified participant bootstrap of fixed out-of-fold predictions; excludes model-refit uncertainty.",
        "validation_note": "Internal nested cross-validation; no independent external test cohort.",
        "fivefold_sensitivity": {
            "protocol": "five repeats of outer stratified 5-fold; inner stratified 5-fold C/gamma search; training-fold-only 5-fold calibration",
            "results": fivefold_results,
            "mean_auroc": float(np.mean([row["auroc"] for row in fivefold_results])),
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fivefold_results])),
            "note": "Exploratory sensitivity check prompted by LOOCV score behavior; repeats reuse the same participants and are not independent validation cohorts.",
        },
        "numeric_note": "macOS Accelerate matmul warnings filtered after finite-input and explicit-dot agreement checks.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("encode", "legacy-evaluate"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--features", type=Path, default=RUN_DIR / "participant_features.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.phase == "encode":
        encode(args.output or RUN_DIR, args.batch_size)
    else:
        evaluate_averaged_baseline(args.output or RBF_RUN_DIR, args.features)
