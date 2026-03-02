#!/usr/bin/env python3
"""
Step 2: Train DAS on Country prompts; find best layer/dim; report IIA.

Modes
-----
  filter  – filter step1_accuracy.jsonl → country_data.jsonl / continent_data.jsonl,
             re-verify accuracy, split country_data 90/10.
  train   – train DAS rotation on country_train, sweep layers × k dims,
             evaluate IIA on country_test and continent_data.
  test    – load a saved DAS checkpoint and evaluate IIA on a given dataset.

Interchange intervention design
---------------------------------
Base and source inputs share the same prompt template but differ only in the city
name (the "quantity").  Patching the country subspace at the city entity token
from source → base should cause the model to predict the source's country.
"""

import os
# Must be set before any HuggingFace library is imported.
os.environ["HF_HOME"] = "/vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import argparse
import copy
import json
import random
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# causalab path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "boundlessDAS"))

from causalab.neural.pipeline import LMPipeline
from causalab.neural.featurizers import Featurizer, SubspaceFeaturizer
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.neural.token_position_builder import TokenPosition, get_substring_token_ids
from causalab.neural.pyvene_core.interchange import (
    run_interchange_interventions,
    train_interventions as train_interventions_pyvene,
)
from causalab.experiments.metric import (
    LM_loss_and_metric_fn,
    causal_score_intervention_outputs,
)
from causalab.causal.counterfactual_dataset import CounterfactualDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS  = Path(__file__).parent.parent / "artifacts"
_HF_CACHE  = Path(__file__).resolve().parent.parent.parent / ".hf_cache"
MODEL_NAME = str(_HF_CACHE / "models--meta-llama--Llama-3.1-8B" / "snapshots"
                 / "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
SEED = 42

# ---------------------------------------------------------------------------
# Normalization + matching
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s'\"\.,;:!\-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    for ch in ["\n", ".", ",", '"', "'"]:
        idx = text.find(ch)
        if idx != -1:
            text = text[:idx].strip()
    return text


def is_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not g:
        return False
    if p.startswith(g):
        return True
    return bool(re.search(r"\b" + re.escape(g) + r"\b", p[:30]))


def ravel_checker(neural_output: Dict[str, Any], expected: Any) -> bool:
    """Two-arg checker compatible with causalab's make_causal_metric."""
    pred = neural_output.get("string", "")
    gold = expected.get("string", "") if isinstance(expected, dict) else str(expected)
    return is_match(pred, gold.strip())


# ---------------------------------------------------------------------------
# Minimal CausalModel wrapper
# ---------------------------------------------------------------------------

class RAVELCausalModel:
    """
    Trivial causal model for RAVEL: labels are pre-stored in the dataset.
    label_counterfactual_data is a no-op because the 'label' column already
    exists in every CounterfactualDataset we build.
    """

    def __init__(self):
        self.id = "ravel"

    def label_counterfactual_data(
        self, dataset: CounterfactualDataset, target_variables
    ) -> CounterfactualDataset:
        return dataset


# ---------------------------------------------------------------------------
# Model / pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(max_new_tokens: int = 6) -> LMPipeline:
    """
    Load Llama-3.1-8B from the local .hf_cache snapshot in bfloat16.

    LMPipeline._setup_model loads in float32 then converts (2× peak RAM → OOM).
    Instead we load directly with torch_dtype=bfloat16 + device_map=auto and
    inject the result into a bare LMPipeline object, bypassing _setup_model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # pyvene requires eager attention and disabled KV-cache
    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build LMPipeline without triggering _setup_model (which would reload)
    pipeline = object.__new__(LMPipeline)
    pipeline.model_or_name     = MODEL_NAME
    pipeline.max_new_tokens    = max_new_tokens
    pipeline.max_length        = None
    pipeline.logit_labels      = False
    pipeline.position_ids      = False
    pipeline.use_chat_template = False
    pipeline.padding_side      = "left"
    pipeline._init_extra_kwargs = {}
    pipeline.model     = model
    pipeline.tokenizer = tokenizer
    return pipeline


# ---------------------------------------------------------------------------
# City entity token position
# ---------------------------------------------------------------------------

def make_city_token_position(pipeline: LMPipeline) -> TokenPosition:
    """
    TokenPosition that returns the index of the last token of the city name.
    Works because base/source prompts share the same template suffix after the
    city, so the city token is at a predictable offset from the sequence end.
    """
    def indexer(input_dict: Dict[str, Any]) -> List[int]:
        prompt = input_dict["raw_input"]
        city = input_dict.get("city", "")
        if city:
            try:
                toks = get_substring_token_ids(
                    text=prompt, substring=city,
                    pipeline=pipeline, add_special_tokens=True,
                )
                return [toks[-1]]
            except Exception:
                pass
        # Fallback: last token of prompt
        ids = list(pipeline.load(input_dict)["input_ids"][0])
        return [len(ids) - 1]

    return TokenPosition(indexer, pipeline, id="city_last_token")


# ---------------------------------------------------------------------------
# CounterfactualDataset builder
# ---------------------------------------------------------------------------

def build_counterfactual_dataset(
    pairs: List[Tuple[dict, dict]], id: str = "ravel"
) -> CounterfactualDataset:
    """
    Build a CounterfactualDataset from (base, source) pairs where base and
    source differ only in the city name.

    Each example:
      input              = {"raw_input": base_prompt, "city": ..., "gold": ...}
      counterfactual_inputs[0] = {"raw_input": src_prompt, "city": ..., "gold": ...}
      label              = " <src_gold>"   (expected output after intervention)
    """
    inputs, cf_inputs, labels = [], [], []
    for base_ex, src_ex in pairs:
        inputs.append({
            "raw_input": base_ex["prompt"],
            "city":      base_ex["city"],
            "gold":      base_ex["gold"],
        })
        cf_inputs.append([{
            "raw_input": src_ex["prompt"],
            "city":      src_ex["city"],
            "gold":      src_ex["gold"],
        }])
        labels.append(" " + src_ex["gold"].strip())

    hf_ds = Dataset.from_dict({
        "input":                inputs,
        "counterfactual_inputs": cf_inputs,
        "label":                labels,
    })
    return CounterfactualDataset(dataset=hf_ds, id=id)


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

def build_pairs(
    examples: List[dict],
    n_pairs: int,
    seed: int = SEED,
    require_diff_gold: bool = True,
) -> List[Tuple[dict, dict]]:
    """Sample (base, source) pairs that differ in city; optionally require
    different gold labels so the intervention is always informative."""
    rng = random.Random(seed)
    pool: List[Tuple[dict, dict]] = [
        (base, src)
        for i, base in enumerate(examples)
        for j, src in enumerate(examples)
        if i != j and (not require_diff_gold or base["gold"] != src["gold"])
    ]
    if len(pool) > n_pairs:
        pool = rng.sample(pool, n_pairs)
    return pool


# ---------------------------------------------------------------------------
# IIA evaluation helper
# ---------------------------------------------------------------------------

def evaluate_iia_on_dataset(
    pipeline: LMPipeline,
    dataset: CounterfactualDataset,
    target,          # InterchangeTarget with (possibly trained) featurizer
    key: Tuple,
    batch_size: int = 8,
) -> float:
    """Evaluate IIA using causalab's run_interchange_interventions + scorer."""
    raw_results = {
        key: run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=False,
        )
    }
    eval_result = causal_score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        causal_model=RAVELCausalModel(),
        target_variable_groups=[("country",)],
        metric=ravel_checker,
    )
    return float(eval_result["results_by_key"][key]["avg_score"])


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def plot_iia_heatmap(
    iia_grid: Dict[int, Dict[int, float]],
    layers: List[int],
    k_dims: List[int],
    title: str,
    save_path: Path,
) -> None:
    sorted_layers = sorted(layers)
    sorted_k      = sorted(k_dims)
    matrix = np.zeros((len(sorted_layers), len(sorted_k)))
    for i, layer in enumerate(sorted_layers):
        for j, k in enumerate(sorted_k):
            matrix[i, j] = iia_grid.get(layer, {}).get(k, float("nan"))

    fig, ax = plt.subplots(
        figsize=(max(4, len(sorted_k) * 1.8), max(3, len(sorted_layers) * 0.45))
    )
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0,
                   cmap="RdYlGn", origin="lower")
    for i in range(len(sorted_layers)):
        for j in range(len(sorted_k)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
    ax.set_xticks(range(len(sorted_k)))
    ax.set_xticklabels([f"k={k}" for k in sorted_k])
    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels([f"L{l}" for l in sorted_layers])
    ax.set_xlabel("Subspace dim  k")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="IIA", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {save_path}")


# ---------------------------------------------------------------------------
# MODE: filter
# ---------------------------------------------------------------------------

def run_filter(args) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    src_path = ARTIFACTS / "step1_accuracy.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} not found — run step1 first.")

    all_rows  = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
    country   = [r for r in all_rows if r["attribute"] == "Country"   and r["correct"]]
    continent = [r for r in all_rows if r["attribute"] == "Continent" and r["correct"]]
    print(f"Filtered: {len(country)} country  |  {len(continent)} continent correct examples")

    (ARTIFACTS / "country_data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in country) + "\n")
    (ARTIFACTS / "continent_data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in continent) + "\n")
    print(f"Saved → {ARTIFACTS}/country_data.jsonl + continent_data.jsonl")

    print("\nRe-running model on filtered examples to verify accuracy …")
    pipeline = load_pipeline(max_new_tokens=6)

    for label, examples in [("Country", country), ("Continent", continent)]:
        correct = 0
        prompt_dicts = [{"raw_input": e["prompt"]} for e in examples]
        for i in tqdm(range(0, len(prompt_dicts), args.batch_size), desc=label):
            batch_dicts = prompt_dicts[i:i + args.batch_size]
            out = pipeline.generate(batch_dicts)
            preds = out["string"] if isinstance(out["string"], list) else [out["string"]]
            for ex, pred in zip(examples[i:i + args.batch_size], preds):
                correct += int(is_match(pred, ex["gold"]))
        print(f"  {label}: {correct}/{len(examples)} = {correct / len(examples):.3f}")

    rng = random.Random(SEED)
    shuffled = country[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_split, test_split = shuffled[:split], shuffled[split:]
    (ARTIFACTS / "country_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_split) + "\n")
    (ARTIFACTS / "country_test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in test_split) + "\n")
    print(f"\nSplit country_data → train: {len(train_split)}, test: {len(test_split)}")
    print(f"Saved → {ARTIFACTS}/country_train.jsonl + country_test.jsonl")


# ---------------------------------------------------------------------------
# MODE: train
# ---------------------------------------------------------------------------

def run_train(args) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    def _load(path: Path) -> List[dict]:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    train_data     = _load(ARTIFACTS / "country_train.jsonl")
    test_data      = _load(ARTIFACTS / "country_test.jsonl")
    continent_data = _load(ARTIFACTS / "continent_data.jsonl")
    print(f"country_train: {len(train_data)}  country_test: {len(test_data)}"
          f"  continent: {len(continent_data)}")

    pipeline  = load_pipeline(max_new_tokens=6)
    num_layers = pipeline.model.config.num_hidden_layers
    d_model    = pipeline.model.config.hidden_size

    # Layer grid
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        step = max(1, num_layers // 8)
        layers = sorted(set(list(range(0, num_layers, step)) + [num_layers - 1]))
    print(f"Sweeping {len(layers)} layers: {layers}")
    print(f"Sweeping k values: {args.k_dims}")

    # Fixed evaluation datasets (built once, reused across all (layer, k))
    test_pairs      = build_pairs(test_data,      n_pairs=args.n_eval_pairs, seed=SEED)
    continent_pairs = build_pairs(continent_data, n_pairs=args.n_eval_pairs, seed=SEED + 1)
    test_cfds       = build_counterfactual_dataset(test_pairs,      id="country_test")
    continent_cfds  = build_counterfactual_dataset(continent_pairs, id="continent")
    print(f"Eval pairs — country_test: {len(test_pairs)}  continent: {len(continent_pairs)}")

    city_pos = make_city_token_position(pipeline)

    # Training config (intervention_type="interchange" → DAS)
    train_config = {
        "intervention_type": "interchange",
        "train_batch_size":  args.batch_size,
        "training_epoch":    args.n_epochs,
        "init_lr":           args.lr,
        "log_dir":           str(ARTIFACTS / "logs"),
        "shuffle":           True,
        "memory_cleanup_freq": 50,
    }

    results: Dict = {}
    best_iia, best_layer, best_k = -1.0, -1, -1
    best_ckpt_dir: Path | None = None

    for layer in layers:
        results[layer] = {}
        for k in args.k_dims:
            print(f"\n── Layer {layer}  k={k} ──")

            # 1. Build training CounterfactualDataset
            train_pairs = build_pairs(
                train_data, n_pairs=args.n_train_pairs,
                seed=SEED + layer * 100 + k,
            )
            train_cfds = build_counterfactual_dataset(train_pairs, id="country_train")

            # 2. Build InterchangeTarget for this layer at the city entity token
            residual_targets = build_residual_stream_targets(
                pipeline=pipeline,
                layers=[layer],
                token_positions=[city_pos],
                mode="one_target_per_layer",
            )
            key    = (layer,)
            target = residual_targets[key]

            # 3. Attach a fresh SubspaceFeaturizer (DAS rotation matrix)
            for unit in target.flatten():
                unit.set_featurizer(SubspaceFeaturizer(
                    shape=(d_model, k),
                    trainable=True,
                    id=f"DAS_L{layer}_k{k}",
                ))

            # 4. Train via pyvene (modifies featurizer in-place)
            train_interventions_pyvene(
                pipeline=pipeline,
                interchange_target=target,
                counterfactual_dataset=train_cfds,
                intervention_type="interchange",
                config={**train_config, "DAS": {"n_features": k}},
                loss_and_metric_fn=partial(LM_loss_and_metric_fn, checker=ravel_checker),
            )

            # 5. Evaluate IIA on country_test and continent
            iia_country = evaluate_iia_on_dataset(
                pipeline, test_cfds, target, key, batch_size=args.batch_size,
            )
            iia_cont = evaluate_iia_on_dataset(
                pipeline, continent_cfds, target, key, batch_size=args.batch_size,
            )
            print(f"  IIA country_test={iia_country:.3f}  IIA_continent={iia_cont:.3f}")
            results[layer][k] = {
                "iia_country_test": iia_country,
                "iia_continent":    iia_cont,
            }

            # 6. Checkpoint if best so far
            if iia_country > best_iia:
                best_iia, best_layer, best_k = iia_country, layer, k
                ckpt_dir = ARTIFACTS / "das_best_featurizer"
                ckpt_dir.mkdir(exist_ok=True)
                feat_base = str(ckpt_dir / "featurizer")
                target.flatten()[0].featurizer.save_modules(feat_base)
                best_ckpt_dir = ckpt_dir
                torch.save(
                    {"layer": layer, "k": k, "d_model": d_model,
                     "featurizer_path": feat_base},
                    str(ARTIFACTS / "das_best.pt"),
                )

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Best: layer={best_layer}  k={best_k}  IIA_country={best_iia:.3f}")
    best_cont = results[best_layer][best_k]["iia_continent"]
    print(f"IIA continent at best model = {best_cont:.3f}")

    results["best"] = {
        "layer": best_layer, "k": best_k,
        "iia_country_test": best_iia,
        "iia_continent":    best_cont,
    }

    print(f"\n{'Layer':>6} {'k':>6} {'IIA_country_test':>18} {'IIA_continent':>15}")
    for layer in sorted(l for l in results if l != "best"):
        for k in sorted(results[layer]):
            rc = results[layer][k]["iia_country_test"]
            ri = results[layer][k]["iia_continent"]
            marker = " ★" if (layer == best_layer and k == best_k) else ""
            print(f"{layer:>6} {k:>6} {rc:>18.3f} {ri:>15.3f}{marker}")

    out_path = ARTIFACTS / "das_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {str(l): {str(k): v for k, v in kv.items()}
             for l, kv in results.items()},
            f, indent=2,
        )
    print(f"\nFull results → {out_path}")

    # Heatmaps
    country_grid   = {l: {k: results[l][k]["iia_country_test"] for k in results[l]}
                      for l in layers}
    continent_grid = {l: {k: results[l][k]["iia_continent"]    for k in results[l]}
                      for l in layers}
    plot_iia_heatmap(
        country_grid, layers, args.k_dims,
        title="DAS IIA on country_test (entity token position)",
        save_path=ARTIFACTS / "das_heatmap_country.png",
    )
    plot_iia_heatmap(
        continent_grid, layers, args.k_dims,
        title="DAS IIA on continent (country-trained, entity token position)",
        save_path=ARTIFACTS / "das_heatmap_continent.png",
    )


# ---------------------------------------------------------------------------
# MODE: test
# ---------------------------------------------------------------------------

def run_test(args) -> None:
    ckpt_path = ARTIFACTS / "das_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run --mode train first.")

    ckpt      = torch.load(ckpt_path, map_location="cpu")
    layer_idx = ckpt["layer"]
    k         = ckpt["k"]
    d_model   = ckpt["d_model"]
    feat_base = ckpt.get("featurizer_path",
                          str(ARTIFACTS / "das_best_featurizer" / "featurizer"))
    print(f"Loaded checkpoint: layer={layer_idx}, k={k}, d_model={d_model}")

    pipeline = load_pipeline(max_new_tokens=6)
    city_pos = make_city_token_position(pipeline)

    residual_targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=[layer_idx],
        token_positions=[city_pos],
        mode="one_target_per_layer",
    )
    key    = (layer_idx,)
    target = residual_targets[key]
    featurizer = Featurizer.load_modules(feat_base)
    for unit in target.flatten():
        unit.set_featurizer(featurizer)

    data = [json.loads(l)
            for l in Path(args.test_data).read_text().splitlines() if l.strip()]
    pairs    = build_pairs(data, n_pairs=args.n_eval_pairs, seed=SEED)
    test_cfs = build_counterfactual_dataset(pairs, id="test")
    print(f"Evaluating {len(pairs)} pairs from {Path(args.test_data).name}")

    iia = evaluate_iia_on_dataset(
        pipeline, test_cfs, target, key, batch_size=args.batch_size,
    )
    print(f"IIA = {iia:.3f}")

    # Replot heatmaps from saved results
    results_path = ARTIFACTS / "das_results.json"
    if not results_path.exists():
        print("das_results.json not found — skipping heatmaps.")
        return

    with open(results_path) as f:
        saved = json.load(f)

    layer_keys = [lk for lk in saved if lk != "best"]
    if not layer_keys:
        return

    layers = sorted(int(lk) for lk in layer_keys)
    k_dims = sorted(int(kk) for kk in saved[str(layers[0])].keys())

    country_grid:   Dict[int, Dict[int, float]] = {}
    continent_grid: Dict[int, Dict[int, float]] = {}
    for layer in layers:
        country_grid[layer]   = {}
        continent_grid[layer] = {}
        for k_val in k_dims:
            entry = saved[str(layer)].get(str(k_val), {})
            country_grid[layer][k_val]   = entry.get("iia_country_test", float("nan"))
            continent_grid[layer][k_val] = entry.get("iia_continent",    float("nan"))

    plot_iia_heatmap(
        country_grid, layers, k_dims,
        title="DAS IIA on country_test (entity token position)",
        save_path=ARTIFACTS / "das_heatmap_country.png",
    )
    plot_iia_heatmap(
        continent_grid, layers, k_dims,
        title="DAS IIA on continent (country-trained, entity token position)",
        save_path=ARTIFACTS / "das_heatmap_continent.png",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",          choices=["filter", "train", "test"], default="filter")
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--layers",        type=str,   default=None,
                   help="Comma-separated layer indices (default: ~8 evenly spaced)")
    p.add_argument("--k_dims",        type=int,   nargs="+", default=[32, 128, 512, 2048])
    p.add_argument("--n_train_pairs", type=int,   default=5000) # 1024 or 2048 
    p.add_argument("--n_eval_pairs",  type=int,   default=200)
    p.add_argument("--n_epochs",      type=int,   default=10)
    p.add_argument("--lr",            type=float, default=5e-3)
    p.add_argument("--test_data",     type=str,
                   default=str(ARTIFACTS / "continent_data.jsonl"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.mode == "filter":
        run_filter(args)
    elif args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
