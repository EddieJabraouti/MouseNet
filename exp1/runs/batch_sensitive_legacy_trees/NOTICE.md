These tree results were computed before discovering the batch-axis indexing
defect in the cloned Mouse2Vec encoder. Their input embeddings depend on
inference batch position. They are retained only to trace the investigation;
use the `stable_*` runs under `exp1/runs/regularized_trees/` for current
comparisons. The defect and corrected adapter are documented in
`exp1/README.md` and `exp1/pte_classifier/stable_mouse2vec.py`.
