#!/usr/bin/env python3
"""Step 1: Evaluate Llama-3.1-8B on RAVEL city Country/Continent prompts."""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use env vars so tokens are not committed. Example: export HF_TOKEN=hf_xxx
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE = os.environ.get("HF_HOME", str(Path(__file__).resolve().parent.parent.parent / ".hf_cache"))

if HF_TOKEN:
    login(token=HF_TOKEN)
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TARGET_ATTRIBUTES = {"Country", "Continent"}

# One fixed template per attribute (simple fill-in-the-blank, no JSON/few-shot).
FIXED_TEMPLATES: dict[str, str] = {
    "Continent": "%s is a city in the continent of",
    "Country":   "%s is a city in the country of",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-model_name", default="meta-llama/Llama-3.1-8B")
    p.add_argument("-batch_size", type=int, default=16)
    return p.parse_args()


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
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    if not gold_norm:
        return False
    if pred_norm.startswith(gold_norm):
        return True
    snippet = pred_norm[:30]
    pattern = r"\b" + re.escape(gold_norm) + r"\b"
    return bool(re.search(pattern, snippet))


def build_examples(entities) -> list[dict]:
    # Each (city, attribute) pair appears exactly once.
    print(f"  Using templates: {FIXED_TEMPLATES}")
    seen_cities: set[str] = set()
    examples = []
    for entity in entities:
        city = entity.get("City", "")
        if not city or city in seen_cities:
            continue
        seen_cities.add(city)
        for attribute, template in FIXED_TEMPLATES.items():
            gold = entity.get(attribute, "")
            if not gold:
                continue
            try:
                prompt_text = template % city
            except (TypeError, ValueError):
                prompt_text = template.replace("%s", city)
            examples.append({
                "city": city,
                "attribute": attribute,
                "prompt": prompt_text,
                "gold": gold,
            })
    return examples


def generate_batch(model, tokenizer, prompts, max_new_tokens=6):
    # Feed raw RAVEL templates directly for next-token completion (base model).
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    padded_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(out[padded_len:], skip_special_tokens=True)
        for out in outputs
    ]


def main():
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    ds = load_dataset("hij/ravel", "city_entity", cache_dir=HF_CACHE, token=HF_TOKEN)
    # Combine all splits so every city in the dataset is evaluated.
    entities = [row for split in ds.values() for row in split]
    print(f"  Total entities across all splits: {len(entities)}")

    examples = build_examples(entities)
    print(f"Built {len(examples)} evaluation examples")
    attr_counts = Counter(e["attribute"] for e in examples)
    for attr, cnt in sorted(attr_counts.items()):
        print(f"  {attr}: {cnt}")

    print("\n=== Sample Prompts ===")
    seen_attrs: set[str] = set()
    shown = 0
    for e in examples:
        if shown >= 6:
            break
        if e["attribute"] not in seen_attrs or shown < 4:
            print(f"  [{e['attribute']}] {e['prompt']!r}  →  gold: {e['gold']!r}")
            seen_attrs.add(e["attribute"])
            shown += 1

    print(f"\nLoading model: {args.model_name}")
    print(f"  HF cache → {HF_CACHE}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=HF_CACHE, token=HF_TOKEN
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=HF_CACHE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    print(f"  Loaded on device: {next(model.parameters()).device}")

    all_results = []
    total = len(examples)
    for start in range(0, total, args.batch_size):
        batch = examples[start : start + args.batch_size]
        preds = generate_batch(model, tokenizer, [e["prompt"] for e in batch])
        for ex, pred in zip(batch, preds):
            all_results.append({**ex, "pred": pred, "correct": is_match(pred, ex["gold"])})
        done = min(start + args.batch_size, total)
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total}")

    by_attr: dict[str, list[bool]] = {}
    for r in all_results:
        by_attr.setdefault(r["attribute"], []).append(r["correct"])

    print("\n=== Accuracy by Attribute ===")
    for attr in sorted(by_attr):
        vals = by_attr[attr]
        acc = sum(vals) / len(vals)
        print(f"  {attr:12s}: {acc:.3f}  ({sum(vals)}/{len(vals)})")
    print("  " + "-" * 36)
    n_correct = sum(r["correct"] for r in all_results)
    print(f"  {'Overall':12s}: {n_correct/total:.3f}  ({n_correct}/{total})")

    failures = [r for r in all_results if not r["correct"]]
    mismatches: Counter = Counter()
    for r in failures:
        mismatches[(r["gold"], normalize(r["pred"]))] += 1

    print("\n=== Top Mismatches ===")
    for (gold, pred), cnt in mismatches.most_common(10):
        print(f"  gold={gold!r:25s} pred={pred!r:25s} count={cnt}")

    print("\n=== Sample Failures ===")
    for r in failures[:5]:
        print(f"  Prompt : {r['prompt']!r}")
        print(f"  Gold   : {r['gold']!r}")
        print(f"  Pred   : {r['pred']!r}")
        print()

    out_path = ARTIFACTS_DIR / "step1_accuracy.jsonl"
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_results)} results → {out_path}")


if __name__ == "__main__":
    main()
