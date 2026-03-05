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
# Load HF token from file if present (for gated models)
_hf_token_path = os.path.join(os.path.dirname(__file__), "..", "hf_token.txt")
if os.path.exists(_hf_token_path):
    with open(_hf_token_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                os.environ["HF_TOKEN"] = line
                break

import sys
import argparse
import copy
import json
import pickle
import random
import re
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
# causalab path and partition_graph (for build_adjacency_from_scores)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "causalab"))
_HYPOTHESIS_TESTING = _REPO_ROOT / "Entity_Binding" / "hypothesis_testing"
if str(_HYPOTHESIS_TESTING) not in sys.path:
    sys.path.insert(0, str(_HYPOTHESIS_TESTING))

from partition_graph import build_adjacency_from_scores

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
# causalab uses list[CounterfactualExample] for datasets (CounterfactualDataset was removed)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS  = Path(__file__).parent.parent / "artifacts"
# Repo id; model is downloaded from HuggingFace Hub.
MODEL_NAME = "meta-llama/Llama-3.1-8B"
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
    exists in every dataset we build.
    """

    def __init__(self):
        self.id = "ravel"

    def label_counterfactual_data(
        self, dataset: list, target_variables
    ) -> list:
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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
    pairs: List[Tuple[dict, dict]], id: str = "ravel", target_attribute = "language"
) -> list:
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
            "attribute": base_ex["attribute"],
            "city":      base_ex["city"],
            "gold":      base_ex["gold"],
            "language":  base_ex["languagegold"],
        })
        cf_inputs.append([{
            "raw_input": src_ex["prompt"],
            "attribute": src_ex["attribute"],
            "city":      src_ex["city"],
            "gold":      src_ex["gold"],
            "language":  src_ex["languagegold"],
        }])
        if (target_attribute or "").lower() == (base_ex["attribute"] or "").lower():
            # The base asks about the target attribute, so the intervention should change the output.
            # Language uses "languagegold" (always present in step1_accuracy_processed.jsonl).
            # Other attributes use "gold" when the source row has the same attribute.
            attr_lower = (target_attribute or "").lower()
            data_key = ATTR_TO_DATA_KEY.get(attr_lower) or ATTR_TO_DATA_KEY.get(target_attribute)
            if data_key is None and (src_ex.get("attribute") or "").lower() == attr_lower:
                data_key = "gold"
            if data_key and data_key in src_ex:
                val = src_ex[data_key]
                labels.append(" " + str(val))
            else:
                raise ValueError(
                    f"Cannot get source value for target {target_attribute}: "
                    f"need key {data_key or 'gold'} (languagegold for Language), src keys: {list(src_ex.keys())}"
                )
        else:
            # The base does not ask about the target attribute. Therefore, the intervention does not change the output.
            labels.append(" " + base_ex["gold"])

    hf_ds = Dataset.from_dict({
        "input":                inputs,
        "counterfactual_inputs": cf_inputs,
        "label":                labels,
    })
    return hf_ds.to_list()


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

def build_pairs(
    examples: List[dict],
    n_pairs: int,
    seed: int = SEED,
    require_diff_gold: bool = True,
    target_attribute: str | None = None,
) -> List[Tuple[dict, dict]]:
    """Sample (base, source) pairs that differ in city; optionally require
    different gold labels so the intervention is always informative.
    If target_attribute is set, only include pairs where BOTH base and source
    have attribute == target_attribute (for evaluation on target-only samples)."""
    rng = random.Random(seed)
    pool: List[Tuple[dict, dict]] = [
        (base, src)
        for i, base in enumerate(examples)
        for j, src in enumerate(examples)
        if i != j and (not require_diff_gold or base["gold"] != src["gold"]) # do we need to add a condition that the city is different?
    ]
    if target_attribute is not None:
        target_attr = _normalize_attr(target_attribute)
        pool = [
            (base, src)
            for base, src in pool
            if base.get("attribute") == target_attr and src.get("attribute") == target_attr
        ]
    if len(pool) > n_pairs:
        pool = rng.sample(pool, n_pairs)
    return pool


# Six attributes in the dataset (must match step1_prep.ATTRIBUTES).
ALL_ATTRIBUTES = ["Continent", "Country", "Language", "Latitude", "Longitude", "Timezone"]

# Map target_attribute to the key in step1_accuracy_processed.jsonl
# Language uses "languagegold" (city language, always present); others use "gold" when attribute matches.
ATTR_TO_DATA_KEY = {"language": "languagegold", "Language": "languagegold"}


def _sample_pairs_from_pools(
    base_pool: List[dict],
    src_pool: List[dict],
    n_requested: int,
    require_diff_gold: bool,
    rng: random.Random,
) -> List[Tuple[dict, dict]]:
    """Sample up to n_requested (base, src) pairs without building full O(n²) pool.
    Uses rejection sampling; assumes pools are large enough for the requested count."""
    if not base_pool or not src_pool:
        return []
    result: List[Tuple[dict, dict]] = []
    seen: set = set()  # (base_city, src_city) for dedup
    max_tries = n_requested * 50
    tries = 0
    while len(result) < n_requested and tries < max_tries:
        base = rng.choice(base_pool)
        src = rng.choice(src_pool)
        key = (base["city"], src["city"])
        if key in seen or base is src:
            tries += 1
            continue
        if require_diff_gold and base["gold"] == src["gold"]:
            tries += 1
            continue
        seen.add(key)
        result.append((base, src))
        tries = 0
    return result


def _normalize_attr(attr: str) -> str:
    """Map user-facing attribute names (e.g. language) to dataset keys (e.g. Language)."""
    for a in ALL_ATTRIBUTES:
        if a.lower() == attr.lower():
            return a
    return attr


def build_pairs_weighted(
    examples: List[dict],
    n_pairs: int,
    target_attribute: str,
    high_weight_ratio: float = 5.0,
    seed: int = SEED,
    require_diff_gold: bool = True,
) -> List[Tuple[dict, dict]]:
    """Sample (base, source) pairs with controlled ratios:
    - high_weight_ratio/(1+high_weight_ratio) pairs with base from target_attribute
    - 1/(1+high_weight_ratio) pairs with base from other attributes
    Within each group, source attributes are evenly distributed across all six.

    Expects flat examples from step1_accuracy_filtered.jsonl (city, attribute, prompt, gold).
    Uses indexed sampling instead of O(n²) pair enumeration for efficiency."""
    rng = random.Random(seed)
    target_attr = _normalize_attr(target_attribute)

    by_attr: Dict[str, List[dict]] = {}
    for ex in examples:
        attr = ex.get("attribute", "")
        by_attr.setdefault(attr, []).append(ex)

    frac_high = high_weight_ratio / (1.0 + high_weight_ratio)
    n_high = int(n_pairs * frac_high)
    n_low = n_pairs - n_high

    result: List[Tuple[dict, dict]] = []
    other_attrs = [a for a in ALL_ATTRIBUTES if a != target_attr]
    target_base_pool = by_attr.get(target_attr, [])

    def divvy(n: int, k: int) -> List[int]:
        base, rem = divmod(n, k)
        return [base + (1 if i < rem else 0) for i in range(k)]

    counts_high = divvy(n_high, 6)
    counts_low = divvy(n_low, 6)

    for i, src_attr in enumerate(ALL_ATTRIBUTES):
        src_pool = by_attr.get(src_attr, [])
        n_this = counts_high[i]
        if n_this > 0:
            pairs = _sample_pairs_from_pools(
                target_base_pool, src_pool, n_this, require_diff_gold, rng
            )
            result.extend(pairs)
    result = result[:n_high]
    high_count = len(result)

    for i, src_attr in enumerate(ALL_ATTRIBUTES):
        src_pool = by_attr.get(src_attr, [])
        n_this = counts_low[i]
        if n_this <= 0:
            continue
        n_per_base_attr = max(1, n_this // len(other_attrs))
        for base_attr in other_attrs:
            base_pool = by_attr.get(base_attr, [])
            if not base_pool:
                continue
            pairs = _sample_pairs_from_pools(
                base_pool, src_pool, n_per_base_attr, require_diff_gold, rng
            )
            result.extend(pairs)
            if len(result) - high_count >= n_low:
                break
        if len(result) - high_count >= n_low:
            break
    result = result[: high_count + n_low]

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# IIA evaluation helper
# ---------------------------------------------------------------------------

def evaluate_iia_on_dataset(
    pipeline: LMPipeline,
    dataset: list,
    target,          # InterchangeTarget with (possibly trained) featurizer
    key: Tuple,
    batch_size: int = 8,
    target_variable: str = "language",
) -> float:
    """Evaluate IIA using causalab's run_interchange_interventions + scorer."""
    batches = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    all_strings = []
    for batch in tqdm(batches, desc="IIA eval", unit="batch", dynamic_ncols=False):
        batch_raw = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=batch,
            interchange_target=target,
            batch_size=len(batch),
            output_scores=False,
        )
        strings = batch_raw.get("string", [])
        for item in strings:
            if isinstance(item, list):
                all_strings.extend(item)
            else:
                all_strings.append(item)
    raw_results = {key: {"string": all_strings}}
    eval_result = causal_score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        causal_model=RAVELCausalModel(),
        target_variable_groups=[(target_variable,)],
        metric=ravel_checker,
    )
    return float(eval_result["results_by_key"][key]["avg_score"])


def compute_per_example_scores_ravel(
    raw_results: Dict,
    cf_dataset: list,
    key: Tuple,
) -> List[float]:
    """Compute per-example consistency scores from raw intervention results (RAVEL/DAS).
    cf_dataset entries have 'label' (expected string). Undirected edge (i,j) only if
    both (i->j) and (j->i) are consistent."""
    string_outputs = raw_results[key].get("string", [])
    flattened = []
    for item in string_outputs:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    scores = []
    for idx, output_string in enumerate(flattened):
        if idx < len(cf_dataset):
            expected = cf_dataset[idx].get("label", "")
            expected_str = expected.get("string", str(expected)) if isinstance(expected, dict) else str(expected)
            is_consistent = ravel_checker({"string": output_string}, expected_str)
            scores.append(1.0 if is_consistent else 0.0)
        else:
            scores.append(0.0)
    return scores


def build_graph_das(
    pipeline: LMPipeline,
    target,
    key: Tuple,
    samples: List[dict],
    target_attribute: str,
    batch_size: int,
) -> np.ndarray:
    """Build undirected adjacency matrix from interchange intervention consistency.
    Edge (i,j) exists iff both (i->j) and (j->i) interventions yield correct output.
    Follows partition_graph.py pattern; does not partition."""
    n = len(samples)
    if n <= 1:
        return np.zeros((n, n), dtype=bool)

    # Pairs (i,j) and (j,i) for all i < j, matching partition_graph create_pair_counterfactuals order
    pairs_ordered: List[Tuple[dict, dict]] = []
    pair_indices: List[Tuple[int, int, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_ordered.append((samples[i], samples[j]))
            pair_indices.append((i, j, "ij"))
            pairs_ordered.append((samples[j], samples[i]))
            pair_indices.append((i, j, "ji"))

    cf_dataset = build_counterfactual_dataset(
        pairs_ordered, id="graph_build", target_attribute=target_attribute
    )
    num_batches = (len(cf_dataset) + batch_size - 1) // batch_size
    print(f"Building graph: {n} nodes, {len(pairs_ordered)} interventions in {num_batches} batches...")

    all_strings = []
    batches = [cf_dataset[i : i + batch_size] for i in range(0, len(cf_dataset), batch_size)]
    for batch in tqdm(batches, desc="Graph interventions", unit="batch", dynamic_ncols=False):
        batch_raw = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=batch,
            interchange_target=target,
            batch_size=len(batch),
            output_scores=False,
        )
        strings = batch_raw.get("string", [])
        for item in strings:
            if isinstance(item, list):
                all_strings.extend(item)
            else:
                all_strings.append(item)
    raw_results = {key: {"string": all_strings}}
    scores = compute_per_example_scores_ravel(raw_results, cf_dataset, key)
    adj_matrix = build_adjacency_from_scores(scores, pair_indices, n)
    return adj_matrix


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
        for i in tqdm(range(0, len(prompt_dicts), args.batch_size), desc=label, dynamic_ncols=False):
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

    data = _load(ARTIFACTS / "step1_accuracy_processed.jsonl")
    rng = random.Random(SEED)
    shuffled = data[:]
    rng.shuffle(shuffled)

    # split the data into train and test (shuffle first so both splits have all attributes)
    split_idx = int(len(shuffled) * 0.8)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    print(f"train: {len(train_data)}  test: {len(test_data)}")

    pipeline  = load_pipeline(max_new_tokens=6)
    num_layers = pipeline.model.config.num_hidden_layers
    d_model    = pipeline.model.config.hidden_size
    target_attribute = args.target_attribute

    # Layer grid
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        step = max(1, num_layers // 8)
        layers = sorted(set(list(range(0, num_layers, step)) + [num_layers - 1]))
    print(f"Sweeping {len(layers)} layers: {layers}")
    print(f"Sweeping k values: {args.k_dims}")

    # Fixed datasets (built once, reused across all (layer, k))
    n_test_pairs = min(args.n_eval_pairs, 300)  # cap at 300 pairs; test_data is target-only
    test_pairs   = build_pairs(test_data, n_pairs=n_test_pairs, seed=SEED, target_attribute=target_attribute)
    train_pairs     = build_pairs_weighted(
        train_data, n_pairs=args.n_train_pairs,
        target_attribute=target_attribute,
        high_weight_ratio=5.0, seed=SEED,
    )
    if not test_pairs:
        raise ValueError(
            f"No test pairs with both base and source attribute == {target_attribute}. "
            "Data may lack sufficient examples; try shuffling before split (now done) or a different target_attribute."
        )
    test_cfds       = build_counterfactual_dataset(test_pairs,  id="country_test",  target_attribute=target_attribute)
    train_cfds      = build_counterfactual_dataset(train_pairs, id="country_train", target_attribute=target_attribute)
    print(f"Eval pairs — test: {len(test_pairs)}")
    target_attr_norm = _normalize_attr(target_attribute)
    if train_pairs:
        base, src = train_pairs[0]
        print("Example train pair — base:", json.dumps(base, indent=2), "source:", json.dumps(src, indent=2))
    if test_pairs:
        base, src = test_pairs[0]
        print("Example test pair — base:", json.dumps(base, indent=2), "source:", json.dumps(src, indent=2))
    target_attr_norm = _normalize_attr(target_attribute)
    n_high = sum(1 for b, _ in train_pairs if b["attribute"] == target_attr_norm)
    pct = 100 * n_high / len(train_pairs) if train_pairs else 0
    print(f"Train pairs: {len(train_pairs)} (target-relevant: {n_high}, ~{pct:.0f}%)")

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

    layer_k_pairs = [(layer, k) for layer in layers for k in args.k_dims]
    for layer, k in tqdm(layer_k_pairs, desc="Train (layer × k)", dynamic_ncols=False):
        results.setdefault(layer, {})
        print(f"\n── Layer {layer}  k={k} ──")

        # 1. Build InterchangeTarget for this (layer, k) at the city entity token
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
            loss_and_metric_fn=lambda p, m, b, t, sp, sim: LM_loss_and_metric_fn(
                p, m, b, t, ravel_checker, source_pipeline=sp, source_intervenable_model=sim
            ),
        )

        # 5. Evaluate IIA on test set only
        iia_test = evaluate_iia_on_dataset(
            pipeline, test_cfds, target, key, batch_size=args.batch_size,
            target_variable=target_attribute,
        )
        print(f"  IIA test={iia_test:.3f}")
        results[layer][k] = {"iia_country_test": iia_test}

        # 6. Checkpoint if best so far
        if iia_test > best_iia:
            best_iia, best_layer, best_k = iia_test, layer, k
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
    print(f"Best: layer={best_layer}  k={best_k}  IIA test={best_iia:.3f}")

    results["best"] = {
        "layer": best_layer, "k": best_k,
        "iia_country_test": best_iia,
    }

    print(f"\n{'Layer':>6} {'k':>6} {'IIA_test':>12}")
    for layer in sorted(l for l in results if l != "best"):
        for k in sorted(results[layer]):
            rc = results[layer][k]["iia_country_test"]
            marker = " ★" if (layer == best_layer and k == best_k) else ""
            print(f"{layer:>6} {k:>6} {rc:>12.3f}{marker}")

    out_path = ARTIFACTS / "das_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {str(l): {str(k): v for k, v in kv.items()}
             for l, kv in results.items()},
            f, indent=2,
        )
    print(f"\nFull results → {out_path}")

    # Heatmap (test set only)
    test_grid = {l: {k: results[l][k]["iia_country_test"] for k in results[l]}
                 for l in layers}
    plot_iia_heatmap(
        test_grid, layers, args.k_dims,
        title="DAS IIA on test set (entity token position)",
        save_path=ARTIFACTS / "das_heatmap_test.png",
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

    test_data_path = Path(args.test_data or str(ARTIFACTS / "step1_accuracy_processed.jsonl"))
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    data = [json.loads(l)
            for l in test_data_path.read_text().splitlines() if l.strip()]
    target_attribute = getattr(args, "target_attribute", "language")
    target_attr_norm = _normalize_attr(target_attribute)

    # Build test dataset: only samples with attribute == target_attribute (follow partition_graph pattern)
    test_dataset = [r for r in data if (r.get("attribute") or "").strip() == target_attr_norm]
    if not test_dataset:
        raise ValueError(
            f"No samples with attribute == {target_attribute!r} in {test_data_path}. "
            "Cannot build test dataset or graph."
        )
    print(f"Test dataset (target_attribute={target_attribute!r}): {len(test_dataset)} samples")

    # Build graph from interchange intervention consistency (do NOT partition)
    test_results_dir = ARTIFACTS / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    graph_size = min(len(test_dataset), getattr(args, "test_graph_size", 50))
    samples_for_graph = test_dataset[:graph_size]
    print(f"Building graph on {len(samples_for_graph)} samples (test_graph_size={graph_size})...")
    adj_matrix = build_graph_das(
        pipeline=pipeline,
        target=target,
        key=key,
        samples=samples_for_graph,
        target_attribute=target_attribute,
        batch_size=args.batch_size,
    )
    with open(test_results_dir / "test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    with open(test_results_dir / "graph.pkl", "wb") as f:
        pickle.dump(adj_matrix, f)
    # Save metadata so we know graph corresponds to first graph_size samples of test_dataset
    with open(test_results_dir / "test_results_meta.json", "w") as f:
        json.dump({
            "target_attribute": target_attribute,
            "n_test_dataset": len(test_dataset),
            "n_graph_nodes": len(samples_for_graph),
            "test_data_path": str(test_data_path),
        }, f, indent=2)
    print(f"Saved test dataset and graph → {test_results_dir} (test_dataset.pkl, graph.pkl); partition not run.")

    """
    # Test: only pairs where BOTH base and source attribute == target_attribute
    pairs    = build_pairs(
        data, n_pairs=args.n_eval_pairs, seed=SEED,
        target_attribute=target_attribute,
    )
    test_cfs = build_counterfactual_dataset(pairs, id="test", target_attribute=target_attribute)
    print(f"Evaluating {len(pairs)} pairs from {test_data_path.name}")

    iia = evaluate_iia_on_dataset(
        pipeline, test_cfs, target, key, batch_size=args.batch_size,
        target_variable=target_attribute,
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

    test_grid: Dict[int, Dict[int, float]] = {}
    for layer in layers:
        test_grid[layer] = {}
        for k_val in k_dims:
            entry = saved[str(layer)].get(str(k_val), {})
            test_grid[layer][k_val] = entry.get("iia_country_test", float("nan"))

    plot_iia_heatmap(
        test_grid, layers, k_dims,
        title="DAS IIA on test set (entity token position)",
        save_path=ARTIFACTS / "das_heatmap_test.png",
    )
    """

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
    p.add_argument("--n_train_pairs", type=int,   default=3000) # 1024 or 2048 
    p.add_argument("--n_eval_pairs",  type=int,   default=300)
    p.add_argument("--n_epochs",      type=int,   default=5)
    p.add_argument("--lr",            type=float, default=5e-3)
    p.add_argument("--target_attribute", type=str, default="language",
                   help="Target attribute for DAS (default: language)")
    p.add_argument("--test_graph_size", type=int, default=200,
                   help="Max number of samples to use when building test graph (default: 50)")
    p.add_argument("--test_data", type=str, default=None,
                   help="Path to test data JSONL (for test mode; default: step1_accuracy_processed)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
