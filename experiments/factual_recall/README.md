# Factual recall / RAVEL experiment

Entangled factual recall using the **RAVEL** benchmark (`datasets.load_dataset("hij/ravel", ...)`) and **Llama-3.1-8B**, targeting the **Language** attribute with **MDAS** / DAS-style training in **`causalab`**.

## Scripts (`scripts/`)

| Script | Role |
|--------|------|
| **`step1_prep.py`** | Score model on RAVEL attributes; write filtered accuracy / counterfactual JSONL under `artifacts/`. |
| **`model_eval.py`** | Narrower country/continent eval slice → `step1_accuracy.jsonl` (used by `step2_das.py filter`). |
| **`step2_das.py`** | `filter` / `train` / `test` modes for subspace alignment; checkpoints and heatmaps under `artifacts/`. |
| **`step4_classifier.py`** | SAE features + logistic classifier on graph partitions; uses local **`partition_graph_quasi_clique.py`** (same algorithm as entity-binding experiment). |
| **`partition_graph_quasi_clique.py`** | Copied shared quasi-clique partitioner (keep in sync with `experiments/entity_binding/partition_graph_quasi_clique.py` if you edit it). |

## Known gap (“step 3”)

`step4_classifier.py` expects **`test_dataset.pkl`** and **`graph.pkl`** under `artifacts/test_results/` or `artifacts/test_results_das/`. The repo does **not** include a single `step3_*.py` that builds these pickles from `step2_das.py` outputs; authors historically produced them via one-off notebooks or manual scripts. If you need full reproducibility, add a small graph-export script or document the exact command sequence you used.

## Setup

```bash
pip install -e ../../causalab
# Plus transformers, sae_lens, etc. as needed for your environment.
```

Optional: `hf_token.txt` in this directory (gitignored) for gated models.

## Legacy code

The old “capital city” prototype is under **`../../archived/factual_recall/Old Capital Code/`**. The vendored upstream **RAVEL** mirror was moved to **`../../archived/factual_recall/ravel/`**; Hugging Face dataset loading does not require it.
