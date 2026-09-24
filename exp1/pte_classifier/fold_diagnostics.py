"""Post-hoc diagnostics for the frozen-window PLS/SVM evaluation.

Ablations below use the parameters selected in each original outer fold;
their test-set results are explanatory diagnostics, not newly tuned estimates.
"""

import csv
import json
from collections import Counter

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .model import RUN_DIR, SEED
from .window_model import WindowPLSSVM
from .window_pls_svm import C_VALUES, COMPONENTS, OUT_DIR, load_windows, participant_scores


def group_summary(labels, values):
    return {str(label): float(np.mean(values[labels == label])) for label in (0, 1)}


def median_window_scores(scores, ids, people):
    return np.asarray([np.median(scores[ids == sid]) for sid in people])


def main():
    windows_path = RUN_DIR / "window_embeddings.npz"
    z, window_ids, y_windows, people, y_people, demo_windows, demographics = load_windows(windows_path)
    with np.load(windows_path, allow_pickle=False) as data:
        parts = data["trial_type"]
    evaluation = json.loads((OUT_DIR / "evaluation.json").read_text())
    with (OUT_DIR / "outer_predictions.csv").open(newline="") as handle:
        saved = {row["subject_id"]: float(row["median_window_decision_score"]) for row in csv.DictReader(handle)}

    rows = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_number, (train, test) in enumerate(outer.split(people, y_people), 1):
        params = evaluation["folds"][fold_number - 1]
        train_mask = np.isin(window_ids, people[train])
        test_mask = np.isin(window_ids, people[test])
        train_z, test_z = z[train_mask], z[test_mask]
        train_demos, test_demos = demo_windows[train_mask], demo_windows[test_mask]
        train_y, train_ids = y_windows[train_mask], window_ids[train_mask]
        test_ids = window_ids[test_mask]
        test_people, test_y = people[test], y_people[test]
        model = WindowPLSSVM(params["n_components"], params["C"]).fit(
            train_z, train_demos, train_y, train_ids
        )
        window_scores = model.decision_function(test_z, test_demos)
        combined_scores = median_window_scores(window_scores, test_ids, test_people)
        saved_scores = np.asarray([saved[sid] for sid in test_people])
        if not np.allclose(combined_scores, saved_scores, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Fold {fold_number} does not reproduce saved predictions")

        # Mouse-only SVM reuses PLS fitted solely on this fold's training
        # windows. No new hyperparameters are selected on the test fold.
        mouse_train = model.pls_scaler.transform(
            model.pls.transform(model.mouse_scaler.transform(train_z))
        ) / np.sqrt(params["n_components"])
        mouse_test = model.pls_scaler.transform(
            model.pls.transform(model.mouse_scaler.transform(test_z))
        ) / np.sqrt(params["n_components"])
        counts = Counter(train_ids)
        class_people = Counter(y_people[train])
        weights = np.asarray(
            [len(train_z) / (2 * class_people[label] * counts[sid]) for sid, label in zip(train_ids, train_y)]
        )
        mouse_svm = SVC(kernel="rbf", C=params["C"], gamma="scale", cache_size=500)
        mouse_svm.fit(mouse_train, train_y, sample_weight=weights)
        mouse_scores = median_window_scores(mouse_svm.decision_function(mouse_test), test_ids, test_people)

        # Demographics-only diagnostic uses one row per participant.
        demo_scaler = StandardScaler().fit(demographics[train])
        demo_svm = SVC(kernel="rbf", C=params["C"], gamma="scale", class_weight="balanced")
        demo_svm.fit(demo_scaler.transform(demographics[train]), y_people[train])
        demo_scores = demo_svm.decision_function(demo_scaler.transform(demographics[test]))

        neutral_demographics = np.tile(model.demo_scaler.mean_, (len(test_z), 1))
        demo_neutral_scores = participant_scores(
            model, test_z, neutral_demographics, test_ids, test_people
        )
        neutral_projected = np.tile(model.pls_scaler.mean_, (len(test_z), 1))
        mouse_neutral_windows = model.svm.decision_function(
            model._combine(neutral_projected, test_demos)
        )
        mouse_neutral_scores = median_window_scores(mouse_neutral_windows, test_ids, test_people)

        part_auc = {}
        for part in ("A", "B"):
            part_mask = parts[test_mask] == part
            part_scores = median_window_scores(window_scores[part_mask], test_ids[part_mask], test_people)
            part_auc[part] = float(roc_auc_score(test_y, part_scores))

        leave_one_out_aucs = [
            float(roc_auc_score(np.delete(test_y, i), np.delete(combined_scores, i)))
            for i in range(len(test))
        ]
        n_positive = int(sum(test_y == 1))
        n_negative = int(sum(test_y == 0))
        window_counts = np.asarray([np.sum(window_ids == sid) for sid in test_people])
        parameter_sensitivity = None
        if fold_number in (1, 3):
            parameter_sensitivity = []
            for components in COMPONENTS:
                for C in C_VALUES:
                    if (components, C) == (params["n_components"], params["C"]):
                        scores = combined_scores
                    else:
                        alternative = WindowPLSSVM(components, C).fit(
                            train_z, train_demos, train_y, train_ids
                        )
                        scores = participant_scores(
                            alternative, test_z, test_demos, test_ids, test_people
                        )
                    parameter_sensitivity.append(
                        {"n_components": components, "C": C, "test_auroc_posthoc": float(roc_auc_score(test_y, scores))}
                    )

        bootstrap_interval = None
        if fold_number == 1:
            rng = np.random.default_rng(SEED)
            positive, negative = combined_scores[test_y == 1], combined_scores[test_y == 0]
            bootstrap_aucs = []
            bootstrap_labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
            for _ in range(5000):
                sample_scores = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
                bootstrap_aucs.append(roc_auc_score(bootstrap_labels, sample_scores))
            bootstrap_interval = [float(v) for v in np.quantile(bootstrap_aucs, (0.025, 0.975))]
        rows.append(
            {
                "fold": fold_number,
                "n_control": n_negative,
                "n_mci": n_positive,
                "selected_pls_components": params["n_components"],
                "selected_C": params["C"],
                "inner_mean_auroc": params["inner_mean_auroc"],
                "combined_auroc": float(roc_auc_score(test_y, combined_scores)),
                "combined_balanced_accuracy": float(balanced_accuracy_score(test_y, combined_scores >= 0)),
                "correctly_ranked_control_mci_pairs": int(round(roc_auc_score(test_y, combined_scores) * n_positive * n_negative)),
                "total_control_mci_pairs": n_positive * n_negative,
                "remove_one_test_participant_auroc_range": [min(leave_one_out_aucs), max(leave_one_out_aucs)],
                "mouse_only_auroc_posthoc": float(roc_auc_score(test_y, mouse_scores)),
                "demographics_only_auroc_posthoc": float(roc_auc_score(test_y, demo_scores)),
                "demographic_block_neutralized_auroc_posthoc": float(roc_auc_score(test_y, demo_neutral_scores)),
                "mouse_block_neutralized_auroc_posthoc": float(roc_auc_score(test_y, mouse_neutral_scores)),
                "age_alone_auroc_descriptive": float(roc_auc_score(test_y, demographics[test, 1])),
                "combined_part_A_auroc_posthoc": part_auc["A"],
                "combined_part_B_auroc_posthoc": part_auc["B"],
                "parameter_sensitivity_posthoc": parameter_sensitivity,
                "fold1_fixed_score_bootstrap_95pct": bootstrap_interval,
                "age_mean_by_group": group_summary(test_y, demographics[test, 1]),
                "education_mean_by_group": group_summary(test_y, demographics[test, 2]),
                "sex_code_1_rate_by_group": group_summary(test_y, demographics[test, 0]),
                "windows_mean_by_group": group_summary(test_y, window_counts),
                "score_mean_by_group": group_summary(test_y, combined_scores),
                "participants": [
                    {
                        "id": str(sid),
                        "group": int(label),
                        "score": float(score),
                        "age": float(demo[1]),
                        "education": float(demo[2]),
                        "sex_code": int(demo[0]),
                        "windows": int(count),
                    }
                    for sid, label, score, demo, count in zip(
                        test_people, test_y, combined_scores, demographics[test], window_counts
                    )
                ],
            }
        )
        print(f"Diagnosed fold {fold_number}/5", flush=True)

    result = {
        "method": "Reproduce each original outer model; refit mouse-only and demographics-only SVMs at its selected parameters; compare post-classifier A/B scores and held-out participant composition.",
        "caution": "All ablations were chosen after inspecting the original folds. They diagnose this split but are not independent performance estimates or causal attributions.",
        "folds": rows,
    }
    (OUT_DIR / "fold_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = [{k: v for k, v in row.items() if k not in ("participants",)} for row in rows]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
