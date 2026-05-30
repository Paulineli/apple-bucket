# Bucketing the Good Apples — code release

Repository for **“Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction”** (LaTeX sources under [`ArXiv/`](ArXiv/)).

The implementation builds on **[causalab](https://github.com/goodfire-ai/causalab)** ([`causalab/`](causalab/) git submodule, pinned to the commit used during this project).

## Layout

| Path | Description |
|------|-------------|
| [`causalab/`](causalab/) | Submodule: intervention utilities, pipelines, entity-binding task definitions. |
| [`experiments/logic_task/`](experiments/logic_task/) | §Toy logic / GPT-2 fine-tuned experiment: DAS, graph partition, natural + SAE classifiers, plots. |
| [`experiments/entity_binding/`](experiments/entity_binding/) | §Entity binding (Gemma-2-2B): step-structured DAS, partition, classifier train/test pipeline. |
| [`experiments/factual_recall/`](experiments/factual_recall/) | §RAVEL / entangled factual recall (Llama-3.1-8B): prep, DAS/MDAS, SAE classifier. |
| [`notebooks/`](notebooks/) | Cross-experiment notebooks. `pca_baselines.ipynb`: supplementary PCA-clustering baseline (see below). |
| [`ArXiv/`](ArXiv/) | Paper `main.tex`, figures, bibliography. |

Authors may keep legacy trees and side experiments under **`archived/`** (only `archived/README.md` and `archived/.gitignore` are tracked; all other paths under `archived/` stay local via `archived/.gitignore`).

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd Multi_Variable_Causal_Abstraction

# Recommended: editable install of causalab (Python ≥ 3.10)
pip install -e ./causalab

# Per experiment (examples)
pip install -r experiments/logic_task/requirements.txt
pip install -r experiments/entity_binding/requirements.txt
```

If you clone without submodules: `git submodule update --init --recursive`.

For gated models, put a Hugging Face token in `experiments/factual_recall/hf_token.txt` (ignored by git).

## Experiment entry points

1. **Logic task (`experiments/logic_task/`)**  
   - Step 1 (train / evaluate DAS): `step1_das.py`  
   - Step 2 (interchange graph + quasi-clique partition): `step2_partition.py`  
   - Step 3 (natural + SAE logistic classifiers): `step3_train_classifier.py`  
   - Step 4 (optional SAE analysis helper): `step4_analyze.py`  
   - Figures: `plot.ipynb`, `Analyze_results.ipynb`

2. **Entity binding (`experiments/entity_binding/`)**  
   - Step 1 (DAS/interchange): `step1_run_das.py`  
   - Step 2 (graph partition): `step2_partition.py` (uses `partition_graph_quasi_clique.py`)  
   - Step 3 (classifier train): `step3_train_classifier.py`  
   - Step 4 (classifier test): `step4_test_classifier.py`, `step4_test_query_group_classifier.py`  
   - See [`experiments/entity_binding/README.md`](experiments/entity_binding/README.md).

3. **Factual recall / RAVEL (`experiments/factual_recall/`)**  
   - Pipeline scripts in `scripts/` (`step1_prep.py`, `step2_das.py`, `step4_classifier.py`)  
   - Plot notebook: `notebooks/plot.ipynb`  
   - See [`experiments/factual_recall/README.md`](experiments/factual_recall/README.md) for the gap between MDAS training and graph-based `step4_classifier.py`.

## Supplementary baseline — PCA clustering

[`notebooks/pca_baselines.ipynb`](notebooks/pca_baselines.ipynb) is a self-contained Colab notebook added in response to a reviewer request for a simpler diagnostic baseline. It replaces **Step 2** of the pipeline (interchange graph + quasi-clique partition) with a plain **PCA + K-means** partition on raw residual-stream activations, then compares the two on identical interchange graphs across all four configurations (Logic L5/P78, Logic L7/P77, Entity Binding L15, RAVEL Language L14) using per-bucket IIA and label-agreement metrics (ARI / NMI). Quasi-cliques are recomputed at the paper's stated `γ = 0.98`.

Outputs (committed alongside the notebook): [`pca_baseline_comparison.csv`](notebooks/pca_baseline_comparison.csv), [`pca_baseline_comparison.json`](notebooks/pca_baseline_comparison.json), and the rendered table [`pca_baseline_table.png`](notebooks/pca_baseline_table.png). The notebook does not depend on the `causalab` submodule. The Logic rows use the fine-tuned GPT-2 checkpoint published on the Hugging Face Hub at [`PaulineLi/bucketing-good-apples-gpt2-logic`](https://huggingface.co/PaulineLi/bucketing-good-apples-gpt2-logic) (public; the notebook downloads it automatically).

## Large artifacts

Model weights, checkpoints, and processed JSONL files are kept **in-tree** under each experiment directory where committed; regenerate or restore from your own backups if missing after clone.
