from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / "exp1/runs/preprocessing_audit/trial_audit.csv"
DEMOGRAPHICS = ROOT / "data/nm9xy-osfstorage-processed_data-archive/demographic_df.csv"
OUTPUT = ROOT / "exp1/runs/TimeCog_net"
SEEDS = tuple(range(42, 52))
SCALES = (1.0, 1.5, 2.0, 4.0)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DEMOGRAPHICS.open(newline="", encoding="utf-8-sig") as handle:
        people = {row["subject_id"]: int(row["group"]) for row in csv.DictReader(handle)}
    if len(people) != 74 or set(people.values()) != {0, 1}:
        raise ValueError("Expected 74 participants in two groups")

    trials: dict[str, dict[int, float]] = {subject_id: {} for subject_id in people}
    with TRIALS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            subject_id = row["subject_id"]
            order = int(row["trial_order"])
            if subject_id not in trials or order not in range(20) or order in trials[subject_id]:
                raise ValueError("Unexpected or duplicate TMT trial")
            if int(row["group"]) != people[subject_id]:
                raise ValueError("Group label mismatch")
            if row["trial_type"] != ("A" if order % 2 == 0 else "B"):
                raise ValueError("Unexpected A/B trial order")
            trials[subject_id][order] = float(row["duration_s"])

    subject_ids = np.asarray(sorted(people))
    if any(set(trials[subject_id]) != set(range(20)) for subject_id in subject_ids):
        raise ValueError("Every participant must have 20 ordered trials")
    durations = np.asarray([[trials[subject_id][order] for order in range(20)] for subject_id in subject_ids])
    groups = np.asarray([people[subject_id] for subject_id in subject_ids], dtype=int)
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Invalid trial duration")
    return subject_ids, groups, durations


class TimeCogNet:
    def __init__(self, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=2,
            gamma=0.5,
            reg_lambda=10,
            reg_alpha=1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        )

    @staticmethod
    def check_input(durations: np.ndarray) -> np.ndarray:
        values = np.asarray(durations, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Expected 20 positive ordered trial durations per participant")
        return values

    def fit(self, durations: np.ndarray, groups: np.ndarray) -> TimeCogNet:
        values = self.check_input(durations)
        labels = np.asarray(groups, dtype=int)
        if labels.shape != (len(values),) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Expected one binary group label per participant")
        self.model.fit(values, labels)
        return self

    def predict_proba(self, durations: np.ndarray, task_scale: float = 1.0) -> np.ndarray:
        if not np.isfinite(task_scale) or task_scale <= 0:
            raise ValueError("Task scale must be positive")
        values = self.check_input(durations) / task_scale
        return self.model.predict_proba(values)

    def comparison_features(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        baseline_task_scale: float = 1.0,
        current_task_scale: float = 1.0,
    ) -> np.ndarray:
        if len(self.check_input(baseline)) != 1 or len(self.check_input(current)) != 1:
            raise ValueError("Comparison requires one complete 20-trial window on each side")
        baseline_score = self.predict_proba(baseline, baseline_task_scale)[0, 1]
        current_score = self.predict_proba(current, current_task_scale)[0, 1]
        return np.asarray([[baseline_score, current_score]])

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> TimeCogNet:
        instance = cls()
        instance.model = joblib.load(path)
        return instance


def evaluate(output: Path = OUTPUT) -> dict:
    subject_ids, groups, durations = load_data()
    predictions = []
    repeats = []
    for seed in SEEDS:
        folds = []
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(subject_ids, groups), 1):
            model = TimeCogNet(random_state=seed).fit(durations[train], groups[train])
            scores = model.predict_proba(durations[test])[:, 1]
            for scale in SCALES:
                shifted = model.predict_proba(durations[test] * scale, task_scale=scale)[:, 1]
                if not np.allclose(scores, shifted, atol=1e-8, rtol=0):
                    raise AssertionError("Known uniform task scale changed predictions")
            folds.append({"fold": fold, "auroc": float(roc_auc_score(groups[test], scores))})
            predictions.extend((seed, fold, subject_id, int(groups[index]), float(score))
                               for index, subject_id, score in zip(test, subject_ids[test], scores))
        repeats.append({"seed": seed, "folds": folds, "mean_fold_auroc": float(np.mean([fold["auroc"] for fold in folds]))})

    means = [repeat["mean_fold_auroc"] for repeat in repeats]
    report = {
        "model": "TimeCogNet XGBoost on 20 ordered TMT trial durations",
        "protocol": "Ten stratified five-fold participant-held-out splits, seeds 42-51; fixed model settings; no synthetic training data",
        "participants": len(subject_ids),
        "trials_per_participant": 20,
        "mean_fold_auroc_across_splits": float(np.mean(means)),
        "seed42_mean_fold_auroc": means[0],
        "repeat_mean_range": [float(np.min(means)), float(np.max(means))],
        "known_uniform_scales_checked": SCALES,
        "trial_data_sha256": file_hash(TRIALS),
        "demographics_sha256": file_hash(DEMOGRAPHICS),
        "repeats": repeats,
        "limitation": "The scale check reverses a known synthetic multiplier. It does not validate predictions on different real tasks.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "fold", "subject_id", "group", "score"))
        writer.writerows(predictions)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(output: Path = OUTPUT) -> None:
    _, groups, durations = load_data()
    output.mkdir(parents=True, exist_ok=True)
    TimeCogNet().fit(durations, groups).save(output / "model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "train"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "evaluate":
        print(json.dumps(evaluate(args.output)["mean_fold_auroc_across_splits"]))
    else:
        train(args.output)
