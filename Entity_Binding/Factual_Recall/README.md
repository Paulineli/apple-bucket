# Factual Recall: Capital Knowledge Alignment

Find where an LLM stores knowledge about **country/state/province capitals** using the causalab package.

## Task

- **Template:** `"The capital of A is B."` We ask the model to fill in **B**.
- **A** = region (country, state, or province); **B** = capital (expected answer).
- **Counterfactual:** Base `(A, B)` vs counterfactual `(A', B')` with `A' ≠ A`.
- **IIA:** Interchange intervention at the **token position of A**: replace base representation at A with counterfactual’s representation at A'. Success = model outputs B' (counterfactual’s capital).
- **Heatmap:** IIA by layer (and optionally by token position), like `replicate.py`.

## Files

- **`config.py`** — `CapitalTaskConfig`: prompt template and `(region, capital)` pairs (default list of countries, US states, Canadian provinces).
- **`causal_models.py`** — `create_capital_model(config)`: causal model with inputs A, B and computed `raw_input`, `raw_output` (same idea as positional entity model).
- **`run_capital_alignment.py`** — Full pipeline:
  1. Build capital causal model.
  2. Generate counterfactual pairs (base vs counterfactual).
  3. Filter: keep only examples where both base and counterfactual answers are correct (like `test_alignment.py`).
  4. Compute IIA at **position A** across layers (residual stream).
  5. Plot heatmap (like `replicate.py`).

## Usage

From `Entity_Binding/hypothesis_testing/Factual_Recall`:

```bash
# Default model Qwen/Qwen3-8B, 128 pairs, batch 32
python run_capital_alignment.py

# Choose intervention token position
python run_capital_alignment.py --token-position A
python run_capital_alignment.py --token-position last_token

# Custom model and sizes
python run_capital_alignment.py --model meta-llama/Llama-3.2-1B-Instruct --dataset-size 128 --batch-size 32

# GPU and output
python run_capital_alignment.py --model Qwen/Qwen3-8B --gpu 0 --output-dir ./my_results

# HuggingFace cache
python run_capital_alignment.py --hf-cache-dir /path/to/cache
```

Results are written under `results/` (or `--output-dir`):

- `datasets/filtered_capital_dataset.json` — filtered counterfactual examples.
- `results/heatmaps/` — IIA heatmap (layer × position A).
- `results/scores.json`, `metadata.json` — scores and run metadata.

## Token position (position of A)

Token positions are built with causalab’s `build_token_position_factories`:

- **Template:** `"The capital of {A} is "` (from `CapitalTaskConfig.prompt_template`).
- **Modes (via `--token-position`)**:
  - **`A`**: last token within variable **A** via scoped index  
    `{"type": "index", "position": -1, "scope": {"variable": "A"}}`
  - **`last_token`**: last token of the full prompt (replicate.py style)  
    `{"type": "index", "position": -1}`

This gives the position of the region name (A) for interchange interventions.

## Dependencies

Uses the **causalab** package under `Entity_Binding/causalab`. The script adds that path automatically. You need:

- `torch`, `transformers`, `pyvene`, `datasets`, `tqdm`, etc. (as in the rest of the repo).
