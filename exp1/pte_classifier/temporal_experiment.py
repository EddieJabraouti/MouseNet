"""Participant-level sequence head on frozen Mouse2Vec window embeddings.

One participant is one training example. All scaling and fitting happens
inside participant-held-out folds. The two GRUs compress ordered windows into
trial states and ordered trial states into a participant state.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model import DEMOGRAPHICS, ROOT, RUN_DIR, SEED, sha256
from .temporal_model import TemporalHead
from .window_pls_svm import load_windows


OUT_DIR = ROOT / "exp1/runs/frozen_mouse2vec_temporal"
HIDDEN_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.02
BATCH_SIZE = 12
SEEDS = (42, 43, 44)
PRIMARY_SEED = 42


def load_sequences(path: Path):
    _, window_ids, _, people, labels, _, demographics = load_windows(path)
    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        order = data["trial_order"]
        parts = data["trial_type"]
        starts = data["window_start_s"]
    max_steps = max(int(np.sum((window_ids == sid) & (order == trial))) for sid in people for trial in range(20))
    windows = np.zeros((len(people), 20, max_steps, 128), dtype=np.float32)
    lengths = np.zeros((len(people), 20), dtype=np.int64)
    for person_index, sid in enumerate(people):
        for trial in range(20):
            indices = np.flatnonzero((window_ids == sid) & (order == trial))
            indices = indices[np.argsort(starts[indices], kind="stable")]
            expected = "A" if trial % 2 == 0 else "B"
            if len(indices) < 1 or set(parts[indices]) != {expected}:
                raise ValueError(f"Missing or mislabeled trial {trial} for {sid}")
            if np.any(np.diff(starts[indices]) <= 0):
                raise ValueError(f"Non-increasing window times for {sid}, trial {trial}")
            lengths[person_index, trial] = len(indices)
            windows[person_index, trial, : len(indices)] = embeddings[indices]
    if int(lengths.sum()) != len(embeddings) or max_steps != 22:
        raise ValueError("Sequence assembly lost or duplicated windows")
    return people, labels, demographics.astype(np.float32), windows, lengths


def standardize_fold(windows: np.ndarray, lengths: np.ndarray, demographics: np.ndarray, train: np.ndarray):
    valid = np.arange(windows.shape[2])[None, None, :] < lengths[:, :, None]
    train_windows = windows[train][valid[train]]
    mean = train_windows.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_windows.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1
    z = (windows - mean) / scale
    z[~valid] = 0
    demo_mean = demographics[train].mean(axis=0)
    demo_scale = demographics[train].std(axis=0)
    demo_scale[demo_scale == 0] = 1
    demo_z = (demographics - demo_mean) / demo_scale
    if not np.isfinite(z).all() or not np.isfinite(demo_z).all():
        raise ValueError("Nonfinite fold-local standardized features")
    scaling = {
        "embedding_mean": torch.from_numpy(mean),
        "embedding_scale": torch.from_numpy(scale),
        "demographic_mean": torch.from_numpy(demo_mean),
        "demographic_scale": torch.from_numpy(demo_scale),
    }
    return torch.from_numpy(z), torch.from_numpy(demo_z), scaling


def train_head(windows, lengths, demographics, labels, train, use_demographics: bool, seed: int, epochs: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalHead(HIDDEN_SIZE, use_demographics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_labels = labels[train]
    n_positive = int(np.sum(train_labels == 1))
    n_negative = int(np.sum(train_labels == 0))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_negative / n_positive))
    for _ in range(epochs):
        model.train()
        for batch_indices in np.array_split(rng.permutation(train), int(np.ceil(len(train) / BATCH_SIZE))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                windows[batch_indices],
                lengths[batch_indices],
                demographics[batch_indices] if use_demographics else None,
            )
            target = torch.from_numpy(labels[batch_indices].astype(np.float32))
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def scores_for(model, windows, lengths, demographics, indices):
    with torch.inference_mode():
        output = model(
            windows[indices], lengths[indices],
            demographics[indices] if model.use_demographics else None,
        ).numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite temporal model score")
    return output


def fold_metrics(labels: np.ndarray, scores: np.ndarray):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0)),
    }


def age_matched_pair_ranking(labels, ages, scores, fold_numbers, tolerance_years=5):
    correct = total = 0
    for fold in np.unique(fold_numbers):
        where = np.flatnonzero(fold_numbers == fold)
        positive = where[labels[where] == 1]
        negative = where[labels[where] == 0]
        for p in positive:
            for n in negative:
                if abs(float(ages[p]) - float(ages[n])) <= tolerance_years:
                    total += 1
                    correct += float(scores[p] > scores[n]) + 0.5 * float(scores[p] == scores[n])
    return {"correct_pairs": correct, "total_pairs": total, "concordance": float(correct / total)}


def save_head(model, scaling, path: Path, use_demographics: bool, seed: int, epochs: int):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaling": scaling,
            "hidden_size": HIDDEN_SIZE,
            "use_demographics": use_demographics,
            "seed": seed,
            "epochs": epochs,
        }, path,
    )


def evaluate(windows_path: Path, out_dir: Path, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS):
    if PRIMARY_SEED not in seeds:
        raise ValueError(f"Primary seed {PRIMARY_SEED} must be included in seeds")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    people, labels, demographics, windows, lengths_np = load_sequences(windows_path)
    lengths = torch.from_numpy(lengths_np)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(people)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_numbers = np.zeros(n, dtype=np.int8)
    scores = {"mouse": np.full(n, np.nan), "demographics": np.full(n, np.nan), "combined": np.full(n, np.nan)}
    seed_scores = {arm: {seed: np.full(n, np.nan) for seed in seeds} for arm in ("mouse", "combined")}
    fold_results = []
    for fold_number, (train, test) in enumerate(outer.split(people, labels), 1):
        if set(people[train]) & set(people[test]):
            raise AssertionError("Participant overlap")
        fold_numbers[test] = fold_number
        z, demo_z, _ = standardize_fold(windows, lengths_np, demographics, train)
        demo_model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", solver="liblinear"))
        demo_model.fit(demographics[train], labels[train])
        scores["demographics"][test] = demo_model.decision_function(demographics[test])
        training_auroc = {"demographics": float(roc_auc_score(labels[train], demo_model.decision_function(demographics[train])))}
        for arm, use_demographics in (("mouse", False), ("combined", True)):
            for seed in seeds:
                model = train_head(z, lengths, demo_z, labels, train, use_demographics, seed, epochs)
                seed_scores[arm][seed][test] = scores_for(model, z, lengths, demo_z, test)
                if seed == PRIMARY_SEED:
                    training_auroc[arm] = float(roc_auc_score(labels[train], scores_for(model, z, lengths, demo_z, train)))
            scores[arm][test] = seed_scores[arm][PRIMARY_SEED][test]
        fold_results.append(
            {
                "fold": fold_number,
                "n_controls": int(np.sum(labels[test] == 0)),
                "n_mci": int(np.sum(labels[test] == 1)),
                "age_mean_controls": float(np.mean(demographics[test][labels[test] == 0, 1])),
                "age_mean_mci": float(np.mean(demographics[test][labels[test] == 1, 1])),
                "scores": {arm: fold_metrics(labels[test], scores[arm][test]) for arm in scores},
                "training_auroc_descriptive": training_auroc,
                "seed_auroc": {
                    arm: {str(seed): float(roc_auc_score(labels[test], seed_scores[arm][seed][test])) for seed in seeds}
                    for arm in ("mouse", "combined")
                },
            }
        )
        print(f"Finished temporal outer fold {fold_number}/5", flush=True)
    for arm in scores:
        if not np.isfinite(scores[arm]).all():
            raise ValueError(f"Missing {arm} out-of-fold score")

    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "group", "age", "outer_fold", "mouse_score", "demographics_score", "combined_score"))
        writer.writerows(zip(people, labels, demographics[:, 1], fold_numbers, scores["mouse"], scores["demographics"], scores["combined"]))

    # Deployment artifacts: refit the same fixed recipe on all participants.
    all_indices = np.arange(n)
    z, demo_z, scaling = standardize_fold(windows, lengths_np, demographics, all_indices)
    for arm, use_demographics in (("mouse", False), ("combined", True)):
        for seed in seeds:
            model = train_head(z, lengths, demo_z, labels, all_indices, use_demographics, seed, epochs)
            save_head(model, scaling, out_dir / f"final_{arm}_seed{seed}.pt", use_demographics, seed, epochs)
    demo_final = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", solver="liblinear"))
    demo_final.fit(demographics, labels)
    joblib.dump(demo_final, out_dir / "final_demographics.joblib")

    report = {
        "model": "frozen Mouse2Vec; GRU over each trial's ordered windows; GRU over 20 ordered trials; participant-level binary loss",
        "architecture": {"hidden_size": HIDDEN_SIZE, "dropout": 0.25, "trainable_parameters_mouse": sum(p.numel() for p in TemporalHead(HIDDEN_SIZE, False).parameters()), "trainable_parameters_combined": sum(p.numel() for p in TemporalHead(HIDDEN_SIZE, True).parameters())},
        "training": {"epochs_fixed": epochs, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "primary_seed": PRIMARY_SEED, "sensitivity_seeds": [seed for seed in seeds if seed != PRIMARY_SEED], "class_balancing": "training-fold BCE positive weight = n_controls/n_mci"},
        "validation": "one outer stratified 5-fold split by participant; no model/hyperparameter selection on held-out folds; primary seed reported, other seeds shown separately for sensitivity; no embedding or prediction averaging",
        "controls": "demographics-only class-balanced logistic regression; mouse-only same GRU head without demographics; combined GRU head with age, sex, education",
        "n_participants": n,
        "class_counts": dict(Counter(int(v) for v in labels)),
        "n_trials": int(n * 20),
        "n_windows": int(lengths_np.sum()),
        "folds": fold_results,
        "summary": {
            arm: {
                "mean_fold_auroc": float(np.mean([row["scores"][arm]["auroc"] for row in fold_results])),
                "mean_fold_balanced_accuracy": float(np.mean([row["scores"][arm]["balanced_accuracy"] for row in fold_results])),
                "pooled_oof_auroc_descriptive": float(roc_auc_score(labels, scores[arm])),
                "age_matched_pairs_within_5_years": age_matched_pair_ranking(labels, demographics[:, 1], scores[arm], fold_numbers),
            }
            for arm in scores
        },
        "mouse_features_sha256": sha256(windows_path),
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "cautions": "Exploratory reuse of the same 74-person cohort and outer split as prior experiments. Age-matched pairs are correlated, not independent subjects; they are a diagnostic rather than an external test. No model was selected by this outer result.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "folds": fold_results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=RUN_DIR / "window_embeddings.npz")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    evaluate(args.windows, args.output, args.epochs, tuple(args.seeds))
