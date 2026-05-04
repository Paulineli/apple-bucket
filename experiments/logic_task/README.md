# Logic task (GPT-2 fine-tuned)

Toy Boolean logic task from the paper: fine-tuned GPT-2 small, **DAS** alignment, interchange graph construction, **quasi-clique** partitioning (via shared module in `experiments/entity_binding/scripts/partition_graph_quasi_clique.py`), and **natural vs SAE** linear classifiers.

## Layout (`scripts/` · `notebooks/` · `artifacts/`)

Mirrors **`experiments/factual_recall/`**: runnable Python lives under **`scripts/`**, exploration under **`notebooks/`**, and generated models / graphs / pickles / figures under **`artifacts/`** (git may ignore large blobs; paths in JSON point at `experiments/logic_task/artifacts/...`). Run scripts with working directory **`experiments/logic_task/`**, or rely on **`scripts/paths.py`** (`EXP_ROOT`, `ARTIFACTS`).

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `step1_das.py` | **Step 1**: train / test DAS; candidate layer-position search; saves weights under `artifacts/training_results/`. |
| `step2_partition.py` | **Step 2**: build `graph_dataset` / adjacency, run `quasi_clique_partition`, save JSON + pickles under `artifacts/partition_results_das/`. |
| `step3_train_classifier.py` | **Step 3**: logistic regression on interpretable vs SAE features for bucket prediction. |
| `step4_analyze.py` | **Step 4**: SAE-based cluster analysis (imports helpers from `experiments/entity_binding/scripts/step3_train_classifier.py`). |
| `util_data.py`, `util_model.py` | Dataset / model loading (`artifacts/models/fine_tuned_gpt2_or/`, `artifacts/data/...`). |

## Notebooks (`notebooks/`)

- `plot.ipynb`, `Analyze_results.ipynb` — figures and sweeps; paths assume cwd is **`experiments/logic_task/`** (e.g. `artifacts/partition_results_das/`). Companion JSON under `artifacts/linear_classifiers_das/` records full paths under `experiments/logic_task/artifacts/...`.

## Setup

```bash
pip install -r requirements.txt
pip install -e ../../causalab   # optional; needed only if you wire in causalab-backed tooling
```

Legacy notebooks, stale scripts, and extra TSV variants are under **`../../archived/logic_task/`**.
