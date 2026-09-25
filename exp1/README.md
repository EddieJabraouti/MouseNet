# TimeCog_net

`TimeCog_net/model.py` is the active, self-contained time-only classifier. Each input is one participant's 20 ordered TMT trial completion times: ten A trials alternating with ten B trials. It uses no mouse movements, keystrokes, demographics, or synthetic training examples. The target is the participant's recorded control/MCI group.

The model reads `exp1/runs/preprocessing_audit/trial_audit.csv` and checks its participant IDs and groups against `data/nm9xy-osfstorage-processed_data-archive/demographic_df.csv`. It uses the existing fixed, regularized XGBoost recipe.

From the repository root, run:

```sh
.venv/bin/python -m exp1.TimeCog_net.model evaluate
.venv/bin/python -m exp1.TimeCog_net.model train
```

`evaluate` writes the ten participant-held-out, stratified five-fold split results and participant predictions to `exp1/runs/TimeCog_net/`. The mean fold AUROC is **0.751** across the ten shuffles; the original seed-42 split scored **0.793**. These shuffles reuse the same 74 participants and are not independent test cohorts. `train` saves a final model fitted on all 74 participants to `exp1/runs/TimeCog_net/model.joblib`; its reported evaluation remains the held-out result.

`TimeCogNet.predict_proba(durations, task_scale=...)` accepts a 20-value profile, or a batch of such profiles, and returns two probabilities per participant (control, MCI). The default scale is 1. If all 20 times are multiplied by the same *known* factor, passing that factor divides the times back to the TMT reference scale before prediction. The evaluation checks factors 1, 1.5, 2, and 4 without changing the predictions. This verifies the uniform-scaling calculation, not performance on a different real task. A future task must have a defensible task scale and the same 20-trial input structure before this model can be evaluated there.

For a baseline comparison, `TimeCogNet.comparison_features(baseline, current)` scores two separate, complete 20-trial windows and returns one row containing `[baseline_MCI_score, current_MCI_score]`. Those two numbers can be passed to a later decision model. Optional separate task scales normalize each window before scoring. No decision model or longitudinal change estimate is trained here: the TMT dataset contains only one completed 20-trial session per participant, so it cannot measure how a person's score changes between sessions. The classifier scores are not clinically calibrated probabilities.

Prior Mouse2Vec and augmentation research is retained as [historical notes](HISTORY.md) and saved results under `exp1/runs/`.
