# MouseNet: frozen Mouse2Vec experiments on digital TMT

These experiments use the official pretrained Mouse2Vec checkpoint as a frozen
feature extractor for the local computerized TMT recordings. We compare (1)
individual 128-dimensional window embeddings through supervised partial least
squares (PLS) and an RBF SVM and (2) a small temporal head that compresses
ordered windows and trials into a participant prediction. **No embeddings are
averaged before either supervised model.**

**Current status (24 September 2026):** The earlier saved Mouse2Vec embeddings
and their 0.601 PLS/SVM result were affected by a batch-position indexing
defect in the cloned encoder. They are retained as historical artifacts, not
as reliable performance estimates. New experiments use a batch-invariant
adapter with the same frozen checkpoint weights. The strongest reproducible
signal found so far is the 20 ordered TMT trial durations, which is specific
to the TMT task and is not a generic passive-mouse representation. See the
follow-up results below. Fixed duration-only RF and XGBoost recipes averaged
0.748 and 0.754 fold AUROC over ten participant-fold shuffles; synthetic
augmentation did not show a consistent gain. The best corrected frozen
Mouse2Vec **embedding-only** result on the initial split was 0.666, but
multiple variants were tried on this same small cohort.

**Modeling direction:** Future supervised comparisons use one input per
participant that contains all 20 ordered TMT trials. For timing, that is the
20-duration vector; for mouse features, it requires a representation that
preserves all 20 trials and their order without averaging embeddings. The
earlier window-level classifiers remain historical diagnostics, not the
architecture to extend. Because every participant has the same alternating
A/B sequence, one-hot A/B columns added to a 20-duration row would be
constant across participants and provide no new information.

## Environment

The cloned code and checkpoint are in `mouse2vec/Mouse2Vec/`. Its `env.yaml`
contains Linux x86 CUDA packages and cannot be resolved on this Apple Silicon
Mac. Conda is not installed here, so this run uses `.venv/` with the host's
PyTorch, NumPy, pandas, and scikit-learn plus `einops==0.6.0` installed in the
venv. The full-model checkpoint was saved with an older PyTorch; the loader in
`pte_classifier/model.py` supplies a missing `GELU.approximate` attribute for
compatibility without modifying the checkpoint or its weights.
Run versions: Python 3.13.5, PyTorch 2.8.0, NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.7.1, XGBoost 3.2.0, einops 0.6.0, and joblib 1.5.1.

Recreate the environment on this host with:

```sh
UV_CACHE_DIR=.uv-cache uv venv .venv --python /usr/local/bin/python3.13 --system-site-packages
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python einops==0.6.0
```

## Frozen embedding extraction

Run from the repository root:

```sh
.venv/bin/python -m exp1.pte_classifier.model encode
```

The adapter reads the 20 experimental A/B trials per participant in
`data/nm9xy-osfstorage-raw_data-archive/`, excluding the two practice trials.
It checks that cursor and button arrays align and timestamps increase. It
converts the PsychoPy height-coordinate range to Mouse2Vec's `[0, 1]` screen
coordinates, resamples with a 20 Hz hold-last rule, and forms five-second
windows with a one-second stride. It computes the time, FFT magnitude, and FFT
phase channels expected by the checkpoint. Frequency channels are scaled
**within each window** so extracting an outer-test participant never uses
statistics from other participants. This differs from the released helper's
batch-wide scaling and is recorded in the extraction manifest.

The checkpoint stays in inference mode. Current extraction writes individual
window embeddings with source and trial metadata and an extraction manifest
under `exp1/runs/frozen_mouse2vec/`. Raw filename IDs are used for the
processed-data join; the manifest records the differing internal IDs. A
`participant_features.npz` in that directory is an artifact of the earlier
averaged-feature experiment and is **not used** by the current pipeline.

## Historical PLS/SVM supervised evaluation (batch-sensitive embeddings)

```sh
.venv/bin/python -m exp1.pte_classifier.window_pls_svm
```

PLS is fitted to **all individual training-window embeddings** and their
participant group labels. Its output has 2, 4, or 8 dimensions, selected only
by inner participant-level validation. Standardization before PLS and after
PLS is also fitted only on training windows. The prior requested sex, age, and
years of education are joined by audited participant ID after PLS; they are
standardized on training participants and concatenated with the compressed
mouse representation for the RBF SVM. The SVM uses `gamma='scale'`, and inner
validation selects `C` from 0.1 or 1. Its sample weights give every training
participant equal total influence regardless of how many windows they have,
while also balancing the two participant groups.

Outer stratified five-fold validation holds out whole participants. Inner
stratified three-fold validation also splits by participant and selects the
settings using **participant-level** AUROC. The model scores each held-out
window separately; the participant score is the median of those
post-classifier decision scores, with zero as the fixed class threshold. The
median is applied only after the classifier and never to input embeddings.
Fold AUROCs are averaged for the primary result, since uncalibrated SVM scores
from separately fitted folds need not share an identical scale. The final
model is then selected and fitted on all 74 participants and saved separately.

## Results and interpretation

The frozen encoder produced **26,655** finite 128-dimensional window
embeddings from **1,480** experimental trials across all **74** participants.
The 74 raw filename IDs match the processed metadata IDs; five files have a different
internal `participant` value. As a separate join check, raw trial durations
and processed `non_cut_rt` values agree closely when joined by filename ID
(Pearson correlation 0.972 across 148 A/B summaries; median absolute
difference 126 ms).

The historical window-level PLS/SVM run achieved **0.601 mean fold AUROC** and
**0.612 mean fold balanced accuracy**. Fold AUROCs were 0.786, 0.536, 0.286,
0.667, and 0.729, showing substantial split sensitivity. Its input embeddings
had the batch-position defect described below. Pooled held-out
scores give a descriptive AUROC of 0.587, but the fold mean is the primary
estimate. The final model selected two PLS components and `C=1`. These are
exploratory internal cross-validation results, not an independent test.

Post-hoc fold diagnostics reproduce every saved prediction. Fold 1 contained
seven controls averaging age 67 and eight MCI participants averaging 76.
Age alone ranked this held-out subset at 0.813 AUROC. With the selected fold-1
settings, demographics alone scored 0.732 AUROC and PLS mouse embeddings alone
scored 0.464; the combination scored 0.786 (44 of 56 case-control pairs
correctly ranked). In the fitted combined model, replacing demographics with
their training mean reduced AUROC to 0.429, whereas replacing the mouse block
with its training mean left 0.750. The same PLS/C settings scored 0.536 in
fold 2; trying all six settings after the fact kept fold 1 at 0.768–0.821 and
fold 3 at 0.214–0.286. This points to held-out cohort composition, especially
demographic separation, as a major reason for the high fold-1 result. The
fixed-prediction bootstrap interval for fold 1 was 0.518–0.982. These are
post-hoc diagnostics, not causal attribution or independent validation. The
reproducible analysis and full per-fold data are in
`pte_classifier/fold_diagnostics.py` and
`exp1/runs/window_pls_rbf/fold_diagnostics.json`.

The earlier averaged-embedding logistic and RBF experiments remain in
`exp1/runs/frozen_mouse2vec/` and
`exp1/runs/rbf_mouse2vec_demographics/` for historical inspection, but they
also used the batch-sensitive extraction and discard within-trial variation.
Historical PLS/SVM outputs are in `exp1/runs/window_pls_rbf/`:
`outer_predictions.csv`, `evaluation.json`, and `final_classifier.joblib`.

This pipeline preserves each Mouse2Vec window through PLS and SVM, but **PLS
treats windows as separate observations**. It does not model their order
across the trial. PLS also sees more training examples from participants with
longer recordings; participant balancing is applied to the SVM fit, not the
PLS fit. The published Mouse2Vec code scales FFT channels over a batch,
whereas this adapter scales each window independently to avoid cross-person
preprocessing leakage; that input difference may affect transfer.

## Historical ordered-window temporal experiment (batch-sensitive embeddings)

Run from the repository root:

```sh
.venv/bin/python -m exp1.pte_classifier.temporal_experiment
```

The temporal experiment keeps all 26,655 frozen 128-dimensional windows in
time order. For each of the 74 participants, a GRU compresses the windows
within each of 20 A/B trials to an 8-dimensional trial state. A second GRU
compresses the 20 ordered trial states to a participant state, followed by a
binary classifier. The combined arm adds sex, age, and years of education at
the classifier. The mouse-only arm has the same architecture without those
three values. Demographics-only is a class-balanced logistic regression.
Window scaling and demographic scaling are fitted on training participants
inside each fold. There is no embedding averaging or prediction averaging.
This run used the same batch-sensitive embeddings as the historical PLS/SVM
run; its scores must be interpreted with that limitation.

The comparison uses the same five participant-held-out outer folds as the
PLS/SVM experiment. The architecture and 50-epoch training recipe were fixed
before evaluating the folds. Seed 42 is the reported model; seeds 43 and 44
are recorded separately as an initialization-sensitivity check, not selected
or averaged. The reported mean fold AUROCs are **0.487 mouse-only**, **0.532
demographics-only**, and **0.473 combined**. Mean fold balanced accuracies are
0.525, 0.484, and 0.505, respectively. The combined model's fold AUROCs are
0.304, 0.375, 0.625, 0.500, and 0.563. Its mean fold AUROC is 0.579 for seed
43 and 0.366 for seed 44, illustrating sensitivity to initialization.

The primary model's training-fold AUROC is approximately 0.99–1.00 in every
fold for both mouse-only and combined arms. That large training/held-out gap
indicates overfitting. The temporal head did not improve on the PLS/SVM's
0.601 mean fold AUROC. In the first outer fold, where the prior PLS/SVM
reached 0.786, the temporal combined model reached only 0.304. An auxiliary
same-fold ranking among MCI/control pairs within five years of age was 0.480
for mouse-only and 0.418 for combined across 98 correlated pairs. These are
diagnostics, not independent age-matched test sets.

The code and machine-readable results are in
`pte_classifier/temporal_model.py`,
`pte_classifier/temporal_experiment.py`, and
`exp1/runs/frozen_mouse2vec_temporal/`. The run directory contains
participant-level held-out predictions, per-fold results, and final all-data
fits of the prespecified recipes. Given the held-out results, those final
fits should not be treated as validated predictive models. Reusing this cohort
and split for several experiments makes all comparisons exploratory; an
untouched external cohort would be needed for a final estimate.

Participant-level splitting and reporting are consistent with the [TMT
study](https://www.nature.com/articles/s41598-026-62955-9) and
[TRIPOD+AI guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11025451/).

## Follow-up: corrected encoder, representation audit, and regularized models

The cloned Mouse2Vec encoder's `TFR.forward` in
`mouse2vec/Mouse2Vec/CRT.py` adds modality embeddings with slices such as
`x[:m_token_idx]`. That selects **batch rows**, although `m_token_idx` is a
**token position**. Encoding the same trajectory in all 64 rows produced a
maximum coordinate difference of 5.75 between embeddings. The
`pte_classifier/stable_mouse2vec.py` adapter applies those embeddings along
the token axis without changing the official checkpoint or third-party code;
its repeated-window outputs agreed to within numerical precision regardless
of batch position. Across all 26,655 aligned windows, the median cosine
similarity between old and corrected embeddings was only 0.709. The audit is
reproducible with `python -m exp1.pte_classifier.audit_encoder_batching` and
saved in `exp1/runs/representation_variants/batching_audit.json`. Because
pretraining may also have used the indexing defect, this is an intended-code
repair, **not a guarantee** that the pretrained weights become more useful.

The source TMT cursor is sampled at about 30 Hz (median) and the Mouse2Vec
adapter resamples to 20 Hz, as in the [Mouse2Vec
paper](https://www.collaborative-ai.org/publications/zhang24_chi.pdf). The
20 Hz hold-last version retains a median 99.4% of each raw trial's path
length, with a worst observed trial retaining 85.8%; the rank correlation of
raw and held path lengths is 0.9996. Thus overall path length is mostly
preserved. A larger task-specific loss is **completion time and clicks**:
they are recorded by the raw TMT client but do not enter the frozen encoder
directly. A/B participant-median completion times alone had descriptive
AUROCs 0.639/0.661. The raw/preprocessing audit and 1,480 per-trial records
are in `exp1/runs/preprocessing_audit/`.

Every current supervised comparison holds out whole real participants. The
common outer split is five stratified folds with seed 42 and the same lexical
participant ID ordering as the earlier Mouse2Vec evaluation; we verified the
held-out ID sets match across current runs. Scalers and PLS are fitted only
on the training portion. Tree and logistic recipes use fixed regularized
settings listed in their JSON reports. We report mean held-out fold AUROC;
window-level models take the median **after** window classification to produce
one participant score. No Mouse2Vec embeddings are averaged. Several models
and preprocessing variants were examined on the same 74-person cohort, so
these are exploratory comparisons, not an untouched test estimate.

On corrected hold-last/per-window-FFT Mouse2Vec embeddings, the fixed
PLS-2/RBF SVM with demographics scored 0.590 mean fold AUROC. A conservative
RF with PLS-2 and demographics scored 0.615; XGBoost with the same features
scored 0.643. A direct 128-dimensional RF **without PLS or demographics**
scored 0.666 on this split, suggesting the PLS bottleneck may discard useful
mouse information. Its fold performance still varies, and the direct result
does not establish a general advantage. Scaling FFT spectra across each
completed participant instead of each window did not improve the comparable
XGBoost model (0.643); switching from hold-last to linear 20 Hz interpolation
also gave 0.643. These representation variants are under
`exp1/runs/representation_variants/`, with regularized classifier results in
`exp1/runs/regularized_trees/`.

The raw TMT timing signal was stronger. Keeping all **20 trial durations in
their original order** (ten A and ten B, with no embedding pooling), the
fixed RF scored 0.762 and the fixed XGBoost scored 0.793 on the common outer
split. Across ten different participant-fold shuffles with **unchanged**
recipes, their mean fold AUROCs averaged 0.748 (range 0.712–0.779) and 0.754
(0.728–0.797), respectively. Extending to 60 ordered raw values (duration,
path length, click count per trial) was less stable: repeat means 0.719 RF
and 0.722 XGBoost. Adding each trial's raw duration as a downstream side
channel to PLS-2 mouse windows raised the mouse-only SVM from 0.592 to 0.742
on the original split. The 20-duration RF and XGBoost ranked 76 and 77 of 98
same-fold MCI/control pairs within five years of age correctly; the
demographics-only RF ranked 50 of 98. Those pairs share participants and are
only a post-hoc diagnostic.

The TMT task appears to have an approximately 25-second trial ceiling: median
raw duration was 24.99 seconds. About 39.7% of control trials and 60.6% of
MCI trials lasted at least 24.8 seconds. The duration models can therefore
use task completion/timeout behavior, which is relevant to this cTMT but
will not exist in unstructured everyday mouse use.

The supplied 103 processed TMT features provide a separate task-specific
benchmark. A fixed L2 logistic model scored 0.688 on digital features and
0.668 with demographics on the common split. The combined model's mean
across ten split shuffles was 0.613, versus 0.598 for a demographics-only RF.
These numbers do not reproduce the published paper's nested evaluation
protocol and should not be directly compared with its reported 0.67/0.70.
Their role here is to check whether the raw task signal is stronger than the
frozen representation on this cohort. The corresponding reports are in
`exp1/runs/raw_trial_baselines/`,
`exp1/runs/processed_feature_baselines/`,
`exp1/runs/stability_checks/`, and
`exp1/runs/comparative_diagnostics/`.
The raw-trial run also saves final all-data duration-only RF and XGBoost fits
as exploratory candidates; their measured performance remains the held-out
fold results above.

## Matched mouse, demographic, and time ablation

`python -m exp1.pte_classifier.matched_modalities` compares seven inputs on
the same ten participant-stratified five-fold shuffles (seeds 42–51). Here
**combined** means mouse plus demographics, and **all** means mouse plus
demographics plus time. We also include time alone as the reference. Mouse is
each individual corrected frozen Mouse2Vec window compressed to two PLS
components using only its training fold. Demographics are sex, age, and years
of education; time is the complete ordered vector of 20 raw TMT trial
durations. Participant-level inputs are repeated for that person's windows.
The classifier is fitted on separate windows with equal total training weight
per participant within class; its held-out window **scores** are then reduced
to one participant score by the median. Embeddings are never averaged.

With the same fixed, heavily regularized RF recipe for every input, mean fold
AUROC across the ten shuffles was: mouse **0.570**, demographic **0.596**,
time **0.752**, combined **0.627**, mouse + time **0.755**, demographic + time
**0.755**, and all **0.753**. With the same fixed XGBoost recipe for every
input, the corresponding results were **0.568**, **0.637**, **0.737**,
**0.636**, **0.742**, **0.737**, and **0.731**. The largest average increment
over time alone was only 0.005 AUROC (XGBoost mouse + time). Its paired
repeat mean exceeded time alone in seven of ten shuffles; RF mouse + time did
so in five of ten. These repeats share all 74 participants, so this is a
stability description, not ten independent validation cohorts or evidence
of a reliable gain.

The time-only scores here differ from the earlier 20-duration participant-level
baseline because this ablation holds the **window-level** training setup and
stronger tree regularization fixed across all seven inputs. Likewise the
mouse-only results use PLS-2, unlike the earlier direct-128D RF result. Full
20-trial time features require a completed TMT session and do not represent
what would be available early in a longitudinal stream. The script, all
10,360 held-out participant predictions, fold metrics, input hashes, and
configuration are in `exp1/runs/matched_modalities/`.

## Can stretching TMT time stand in for longer tasks?

`python -m exp1.pte_classifier.time_stretch_probe` tests this assumption as
an input-shift experiment. Each TMT person has 20 trial times; the original
RF/XGBoost duration classifiers are fitted in ten participant-held-out
five-fold shuffles. For held-out people, every one of their 20 times is
multiplied by 1.25, 1.5, 2, or 4. The test also fits models with stretched
**training-only** copies, giving each original participant the same total
training weight. All test participants remain real. The stretched test
profiles are hypothetical transformations of their real recordings, not
observed long tasks or independent clinical cases.

The original time model is not robust to this change. Mean fold AUROC for the
RF was **0.746** on real held-out durations, **0.617** at 1.25×, and **0.500**
at 4×. XGBoost fell from **0.751** to **0.606** and **0.500**. Training on
copies spanning 1–4× reduced original-duration AUROC to **0.698 RF** and
**0.529 XGBoost**; at 4×, those models scored **0.523** and **0.500**. This
augmentation does not establish transfer and, under this recipe, removes
much of the useful TMT time signal.

A separate exploratory model divides each participant's 20 trial times by
that participant's median trial time. This retains all 20 relative values
and is mathematically unchanged by a uniform multiplier. Its original TMT
AUROC was **0.733 RF** and **0.656 XGBoost**, and its predictions were
unchanged at every tested multiplier. This is a possible way to study
*within-task timing pattern* independently of overall speed; it cannot
classify a single long task and does not establish transfer between task
types. It also uses the whole completed TMT session.

The data have no real longer TMT trials on which to verify the stretched
labels: 51.3% of TMT trial durations are at least 24.8 seconds, with a
median of 24.99 seconds and a maximum of 25.97 seconds. The Mendeley
form-filling data contain 1,760 event-spanned sessions with a 41.37-second
median and 94.35-second 90th percentile. They demonstrate a longer-task
distribution, but their tasks differ, their event spans can include idle
time, and they cannot serve as MCI test outcomes. The code, fold results,
22,200 synthetic-shift predictions, and duration-distribution summary are
in `exp1/runs/time_stretch_probe/`. Transfer to the proposed client pipeline
remains **unverified** and requires naturally longer, consistently defined
tasks with prospective outcome evaluation.

### Follow-up: uniform scaling with the task type known

The stress test above gave longer times to a model trained only at the
original TMT scale. That is different from a system that knows the new task
type and its characteristic time multiplier. The controlled follow-up in
`pte_classifier/task_conditioned_stretch.py` explicitly tests that case on
the same ten participant-held-out shuffles. It makes synthetic task types by
uniformly multiplying *both* groups' entire 20-trial profiles by 1, 1.5, 2,
or 4. Separate models trained and tested at each scale retained the original
AUROC at all four scales: **0.746 RF** and **0.751 XGBoost**. Dividing the
times by the known task multiplier before applying the original model gave
identical AUCs and effectively unchanged participant scores. This confirms
that uniform scaling itself need not erase the MCI/control ordering when the
scale is known and handled consistently.

Giving one pooled model the raw scaled times plus a one-hot synthetic task
label was less effective with the current fixed, heavily regularized tree
recipes: at 4× it scored **0.529 RF** and **0.500 XGBoost**. A task label is
therefore not by itself a guarantee that a limited tree learns the needed
task-specific time thresholds. Explicit task normalization or separate
per-task models implement that assumption more directly. The synthetic task
types in this probe differ *only* by a known multiplier; they do not test
real differences in task demands, pausing, timeouts, or the meaning of one
completion time versus the current 20-trial vector. The 23,680 held-out
predictions and full run report are in `exp1/runs/task_conditioned_stretch/`.

## Synthetic TMT experiment

The generator in `pte_classifier/synthetic_tmt.py` made two derived profiles
per real participant: 148 synthetic profiles, 2,960 trials, and 52,623
frozen-encoder windows. It applies a 0.98–1.02 trial-duration scale and small
smooth coordinate jitter; it copies button states and keeps coordinates on
screen. Sex and group stay with the source. Ages are sampled within two years
of the source and 55–90; education stays within one year and 7–20. An audit
checks timestamps, screen bounds, clicks, demographics, and source labels.
Timing and jitter perturbations are standard candidate augmentations for
movement time series, but [the published review](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327038)
does not support assuming that any one transformation is universally valid.
The generated raw trajectories, frozen embeddings, manifest, and audit are
under `exp1/runs/synthetic_tmt/`.

Every synthetic profile is linked to its real source. During evaluation, a
source and all descendants are restricted to training folds together; the
test folds contain **only untouched real participants**. Each source lineage
has equal total classifier training weight, and RF leaf size is scaled with
replica count. PLS itself still sees all augmented windows without lineage
weights, so participants with longer recordings have more influence on that
compression step. For the 20-duration RF, synthetic training changed the
original-split mean fold AUROC from 0.762 to 0.784. Across ten split shuffles,
the means were 0.748 real-only and 0.756 augmented; augmentation helped in 6
of 10 shuffles. That small, inconsistent difference does not establish a gain in
generalization or create new independent clinical labels. Synthetic
trajectory path lengths rose by a median 1.4% despite the small jitter, so
kinematic augmentation warrants particular caution.

The frozen-embedding comparison was similarly mixed on the common real-only
test folds. Synthetic training changed PLS-2 XGBoost with demographics from
0.643 to 0.644, direct 128-dimensional mouse-only RF from 0.666 to 0.625,
and direct 128-dimensional XGBoost with demographics from 0.641 to 0.662.
All fixed-recipe paired results are in
`exp1/runs/regularized_trees/stable_hold_window_direct/` and
`exp1/runs/regularized_trees/stable_hold_window_synthetic/`. The breadth of
models tried and lack of an untouched cohort make small improvements here
particularly easy to overread.

For every current classifier run, we re-read the saved held-out predictions
and reproduced every per-fold AUROC in its JSON report. Every model had one
prediction for each of 74 unique real participants, and the held-out ID sets
matched across the nine current run directories. Code compilation and
`git diff --check` passed. These integrity checks establish reproducibility
of the recorded comparisons; they do not supply external validation.

## References

- [Mouse2Vec paper](https://collaborative-ai.org/publications/zhang24_chi.pdf)
- [Mouse2Vec code and pretrained checkpoint](https://git.cai.simtech.uni-stuttgart.de/public-projects/Mouse2Vec)
- [TMT data](https://osf.io/nm9xy/)
