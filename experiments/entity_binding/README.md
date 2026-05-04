# Entity binding experiment

Positional causal abstraction on the entity-binding task family (paper §Entity Binding), using **`causalab`** task configs and interchange helpers.

## Layout (`scripts/` · `notebooks/` · `artifacts/`)

Same convention as **`experiments/factual_recall/`**: code in **`scripts/`**, Jupyter in **`notebooks/`**, outputs in **`artifacts/`** (graphs, partitions, classifier runs, optional `artifacts/outputs/<model>/` from **`step1_run_das.py`**). Use working directory **`experiments/entity_binding/`** when running **`python scripts/<name>.py`**, or **`scripts/paths.py`** for `EXP_ROOT` / `ARTIFACTS`.

## Pipeline (paper-aligned)

1. **`scripts/step1_run_das.py`** — Filter prompts, run vanilla interchange / alignment search, write metrics and datasets under **`artifacts/outputs/<model>/`** (mkdir on first run; large dirs are often gitignored).
2. **`scripts/step2_partition.py`** — Build the interchangeability graph from saved evaluations and partition with **`scripts/partition_graph_quasi_clique.py`** (quasi-clique method, not spectral clustering).
3. **`scripts/step3_train_classifier.py`** — Train / analyze SAE-based classifiers on bucket membership.
4. **`scripts/step4_test_classifier.py`** / **`scripts/step4_test_query_group_classifier.py`** — Evaluate filtered graphs and IIA using trained classifiers.

Legacy plotting / alignment scripts were moved to **`../../archived/entity_binding/scripts/`**.

## Dependencies

Install the repo-root causalab submodule (`pip install -e ../../causalab` from this folder, or from repo root), then:

```bash
pip install -r requirements.txt
```

## Paper configuration

Main line in the paper: **`filling_liquids`**, **`google/gemma-2-2b-it`**, layer **15**, query last token **10**, partition / classifier artifacts under **`artifacts/partition_results/`** and **`artifacts/classifier_filter_results/`** with `15_10` in the filename.

Legacy READMEs and exploratory runs (other models, PCA line, `15_6` ablations) live under **`../../archived/entity_binding/`** for authors only.
