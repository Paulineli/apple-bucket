#!/usr/bin/env python3
"""Step 1a: Evaluate Llama-3.1-8B on RAVEL city prompts for all 6 attributes; build counterfactual dataset."""

import os
os.environ["HF_HOME"] = "/vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

HF_CACHE  = Path(os.environ["HF_HOME"])
ARTIFACTS = Path(__file__).parent.parent / "artifacts"
SNAPSHOT  = str(HF_CACHE / "models--meta-llama--Llama-3.1-8B"
                          / "snapshots" / "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")

ATTRIBUTES = ["Continent", "Country", "Language", "Latitude", "Longitude", "Timezone"]

FIXED_TEMPLATES = {
    "Continent": "%s is a city in the continent of",
    "Country":   "%s is a city in the country of",
    "Language":  "People in %s speak",
    "Latitude":  "The latitude of %s is",
    "Longitude": "The longitude of %s is",
    "Timezone":  "The timezone of %s is",
}

SEED = 42


# ---------------------------------------------------------------------------
# Normalization + matching (same as model_eval.py / step2_das.py)
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


# ---------------------------------------------------------------------------
# Model loading (bfloat16 + device_map=auto from local snapshot)
# ---------------------------------------------------------------------------

def load_model():
    print(f"  Loading from: {SNAPSHOT}")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 10) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    padded_len = inputs["input_ids"].shape[1]
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(out[padded_len:], skip_special_tokens=True) for out in outputs]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_examples(entities: list[dict], max_per_attr: int, rng: random.Random) -> list[dict]:
    """One example per (city, attribute) pair; cap at max_per_attr per attribute."""
    by_attr: dict[str, list[dict]] = {a: [] for a in ATTRIBUTES}
    seen: set[str] = set()
    for entity in entities:
        city = entity.get("City", "")
        if not city or city in seen:
            continue
        seen.add(city)
        for attr in ATTRIBUTES:
            gold = str(entity.get(attr, "")).strip()
            if not gold:
                continue
            by_attr[attr].append({
                "city": city, "attribute": attr,
                "prompt": FIXED_TEMPLATES[attr] % city,
                "gold": gold,
            })
    for attr in ATTRIBUTES:
        if len(by_attr[attr]) > max_per_attr:
            by_attr[attr] = rng.sample(by_attr[attr], max_per_attr)
    return [ex for attr in ATTRIBUTES for ex in by_attr[attr]]


def evaluate(model, tokenizer, examples: list[dict], batch_size: int) -> list[dict]:
    results = []
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[i:i + batch_size]
        preds = generate_batch(model, tokenizer, [e["prompt"] for e in batch])
        for ex, pred in zip(batch, preds):
            results.append({**ex, "pred": pred, "correct": is_match(pred, ex["gold"])})
    return results


def report_accuracy(results: list[dict], label: str = "") -> None:
    if label:
        print(f"\n=== Accuracy ({label}) ===")
    by_attr: dict[str, list] = {}
    for r in results:
        by_attr.setdefault(r["attribute"], []).append(r["correct"])
    for attr in ATTRIBUTES:
        if attr not in by_attr:
            continue
        vals = by_attr[attr]
        print(f"  {attr:12s}: {sum(vals)/len(vals):.3f}  ({sum(vals)}/{len(vals)})")
    n = sum(r["correct"] for r in results)
    print(f"  {'Overall':12s}: {n/len(results):.3f}  ({n}/{len(results)})")


def build_counterfactual_dataset(
    filtered_by_attr: dict[str, list[dict]],
    entity_lookup: dict[str, dict[str, str]],
    n_per_combo: int,
    rng: random.Random,
) -> list[dict]:
    """
    3600 pairs: 36 (base_attr, source_attr) combos × 100 each.
    Label = source entity's gold value for BASE attribute (DAS target after
    patching source entity representation into base's computation).
    """
    pairs: list[dict] = []
    for base_attr in ATTRIBUTES:
        base_pool = filtered_by_attr.get(base_attr, [])
        for src_attr in ATTRIBUTES:
            src_pool = filtered_by_attr.get(src_attr, [])
            if not base_pool or not src_pool:
                continue
            base_shuf = rng.sample(base_pool, len(base_pool))
            src_shuf  = rng.sample(src_pool,  len(src_pool))
            sampled = 0
            for base_ex in base_shuf:
                if sampled >= n_per_combo:
                    break
                for src_ex in src_shuf:
                    if sampled >= n_per_combo:
                        break
                    if base_ex["city"] == src_ex["city"]:
                        continue
                    # Label: source entity's value for BASE attribute
                    label = entity_lookup.get(src_ex["city"], {}).get(base_attr, "")
                    if not label:
                        continue
                    pairs.append({
                        "base_entity":      base_ex["city"],
                        "base_attribute":   base_attr,
                        "base_prompt":      base_ex["prompt"],
                        "base_gold":        base_ex["gold"],
                        "source_entity":    src_ex["city"],
                        "source_attribute": src_attr,
                        "source_prompt":    src_ex["prompt"],
                        "source_gold":      src_ex["gold"],
                        "label":            label,
                    })
                    sampled += 1
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-max_examples", type=int, default=1000,
                        help="Max examples per attribute (deterministic sample)")
    parser.add_argument("-batch_size",   type=int, default=16)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ── Load RAVEL entities ──────────────────────────────────────────────────
    print("Loading RAVEL city_entity dataset...")
    ds = load_dataset("hij/ravel", "city_entity", cache_dir=str(HF_CACHE))
    entities = [row for split in ds.values() for row in split]
    print(f"  Total entities across all splits: {len(entities)}")

    entity_lookup: dict[str, dict[str, str]] = {
        e["City"]: {a: str(e.get(a, "")).strip() for a in ATTRIBUTES}
        for e in entities if e.get("City")
    }

    # ── Build examples ───────────────────────────────────────────────────────
    print(f"\nBuilding examples (max {args.max_examples} per attribute)...")
    examples = build_examples(entities, max_per_attr=args.max_examples, rng=rng)
    by_attr_all: dict[str, list] = {}
    for ex in examples:
        by_attr_all.setdefault(ex["attribute"], []).append(ex)
    print(f"  Total: {len(examples)}")
    for attr in ATTRIBUTES:
        print(f"    {attr:12s}: {len(by_attr_all.get(attr, []))}")

    print("\n=== Sample Prompts ===")
    for attr in ATTRIBUTES:
        pool = by_attr_all.get(attr, [])
        if pool:
            e = pool[0]
            print(f"  [{attr:12s}] {e['prompt']!r}  →  gold: {e['gold']!r}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading Llama-3.1-8B...")
    model, tokenizer = load_model()
    print(f"  Loaded on device: {next(model.parameters()).device}")

    # ── Evaluate (full) ──────────────────────────────────────────────────────
    results = evaluate(model, tokenizer, examples, batch_size=args.batch_size)
    report_accuracy(results, label="all examples")

    failures = [r for r in results if not r["correct"]]
    mismatches: Counter = Counter((r["gold"], normalize(r["pred"])) for r in failures)
    print("\n=== Top Mismatches ===")
    for (gold, pred), cnt in mismatches.most_common(10):
        print(f"  gold={gold!r:25s} pred={pred!r:25s} count={cnt}")
    print("\n=== Sample Failures ===")
    for r in failures[:5]:
        print(f"  [{r['attribute']:12s}] {r['prompt']!r}")
        print(f"    gold={r['gold']!r}  pred={r['pred']!r}")

    # ── Save all results ─────────────────────────────────────────────────────
    out_all = ARTIFACTS / "step1_accuracy_all.jsonl"
    out_all.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nSaved {len(results)} results → {out_all}")

    # ── Filter to correct-only ───────────────────────────────────────────────
    filtered = [r for r in results if r["correct"]]
    out_filt = ARTIFACTS / "step1_accuracy_filtered.jsonl"
    out_filt.write_text("\n".join(json.dumps(r) for r in filtered) + "\n")
    print(f"Saved {len(filtered)} filtered results → {out_filt}")

    filtered_by_attr: dict[str, list] = {}
    for r in filtered:
        filtered_by_attr.setdefault(r["attribute"], []).append(r)
    report_accuracy(filtered, label="filtered (should be 1.0)")

    # ── Build counterfactual dataset ─────────────────────────────────────────
    print("\nBuilding counterfactual dataset (36 combos × 100 = 3600 pairs)...")
    rng2 = random.Random(SEED + 1)
    cf_pairs = build_counterfactual_dataset(filtered_by_attr, entity_lookup,
                                            n_per_combo=100, rng=rng2)
    combo_counts = Counter((p["base_attribute"], p["source_attribute"]) for p in cf_pairs)
    print(f"  Total pairs: {len(cf_pairs)}")
    print(f"  Combos with <100 pairs: "
          f"{sum(1 for c in combo_counts.values() if c < 100)}/36")
    for (ba, sa), cnt in sorted(combo_counts.items()):
        marker = " !" if cnt < 100 else ""
        print(f"    {ba:12s} × {sa:12s}: {cnt}{marker}")

    out_cf = ARTIFACTS / "step1_counterfactuals.jsonl"
    out_cf.write_text("\n".join(json.dumps(p) for p in cf_pairs) + "\n")
    print(f"\nSaved {len(cf_pairs)} counterfactual pairs → {out_cf}")

    print("\nDone. Please review the results before proceeding to Step 2.")


if __name__ == "__main__":
    main()
