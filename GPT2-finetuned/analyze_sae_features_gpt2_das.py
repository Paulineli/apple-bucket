#!/usr/bin/env python
"""
Analyze SAE features for GPT2 DAS partition results.

For a given layer and position (from partition_results_das JSON), this script:
1. Loads the graph_dataset and partition labels
2. Extracts residual-stream activations at (layer, pos) from GPT2
3. Loads a pretrained GPT2 SAE (e.g. from sae_lens: gpt2-small-res-jb)
4. Encodes activations with the SAE and trains LogisticRegression for cluster classification

Reuses classifier logic from Entity_Binding/hypothesis_testing/analyze_sae_features.py.
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Add GPT2-finetuned for util_data, util_model
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Add Entity_Binding/hypothesis_testing for shared SAE/classifier helpers
_partition_dir = os.path.join(_script_dir, "..", "Entity_Binding", "hypothesis_testing")
if _partition_dir not in sys.path:
    sys.path.insert(0, _partition_dir)

import util_data
import util_model

# Optional: reuse classifier and analysis from analyze_sae_features (avoids causalab)
try:
    from analyze_sae_features import (
        analyze_features_by_cluster,
        train_cluster_classifier,
        train_cluster_classifier_full_l1,
        save_classifier,
        SAE_LENS_AVAILABLE,
        SAE,
    )
except ImportError:
    SAE_LENS_AVAILABLE = False
    SAE = None
    analyze_features_by_cluster = None
    train_cluster_classifier = None
    train_cluster_classifier_full_l1 = None
    save_classifier = None

# Fallback: try sae_lens directly if not from analyze_sae_features
if not SAE_LENS_AVAILABLE or SAE is None:
    try:
        from sae_lens import SAE
        SAE_LENS_AVAILABLE = True
    except ImportError:
        SAE = None
        SAE_LENS_AVAILABLE = False


def _batch_tokenize_texts(tokenizer, texts: List[str], device: str, left_pad: bool = True) -> torch.Tensor:
    """Tokenize a list of texts; return (N, L) left-padded input_ids."""
    if not texts:
        return torch.zeros(0, 0, dtype=torch.long, device=device)
    old_side = getattr(tokenizer, "padding_side", "right")
    if left_pad:
        tokenizer.padding_side = "left"
    max_len = getattr(tokenizer, "model_max_length", 200)
    out = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )["input_ids"].to(device)
    tokenizer.padding_side = old_side
    return out


def extract_activations_at_layer_pos(
    model: torch.nn.Module,
    tokenizer,
    graph_dataset: List[Dict],
    layer: int,
    pos_num: int,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Extract residual-stream activations at (layer, pos_num) for each sample in graph_dataset.

    Uses output_hidden_states; for GPT2, hidden_states[0] is embeddings, hidden_states[i] is
    output of block (i-1). So after block `layer` we use hidden_states[layer+1].
    Position: we use min(pos_num, seq_len-1) per sequence so we don't index out of bounds.

    Returns:
        Tensor of shape (n_samples, hidden_size).
    """
    model.eval()
    texts = []
    for dp in graph_dataset:
        text = util_data.format_input(
            dp["input_ids"],
            dp["context_texts"],
            dp["context_labels"],
        )
        texts.append(text)

    all_activations = []
    n = len(texts)
    for start in tqdm(range(0, n, batch_size), desc="Extracting activations", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        input_ids = _batch_tokenize_texts(tokenizer, batch_texts, device, left_pad=True)
        # (B, L)
        seq_lens = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        # hidden_states: tuple of (embed, layer0_out, ..., layer11_out) -> 13 tensors for GPT2-small
        # We want residual stream *after* block `layer` -> hidden_states[layer+1]
        hidden = outputs.hidden_states[layer + 1]  # (B, L, H)
        batch_size_cur = hidden.shape[0]
        for i in range(batch_size_cur):
            pos = min(int(pos_num), int(seq_lens[i].item()) - 1)
            if pos < 0:
                pos = 0
            all_activations.append(hidden[i, pos, :].cpu())
    return torch.stack(all_activations)


def load_sae_gpt2(layer: int, device: str = "cuda:0", sae_path: str = None):
    """
    Load a pretrained GPT2 SAE for the given layer.

    Default: SAELens release "gpt2-small-res-jb", sae_id "blocks.{layer}.hook_resid_post".
    """
    if not SAE_LENS_AVAILABLE or SAE is None:
        raise ImportError(
            "sae_lens is not available. Install with: pip install sae-lens"
        )
    if sae_path:
        if os.path.exists(sae_path):
            return SAE.load_from_disk(sae_path, device=device)
        if ":" in sae_path:
            release, sae_id = sae_path.split(":", 1)
            try:
                return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            except TypeError:
                sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
                return sae
        return SAE.from_pretrained(sae_path, device=device)
    # Default: gpt2-small-res-jb residual stream SAEs (all layers)
    release = "gpt2-small-res-jb"
    sae_id = f"blocks.{layer}.hook_resid_post"
    try:
        return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    except Exception:
        sae_id_pre = f"blocks.{layer}.hook_resid_pre"
        return SAE.from_pretrained(release=release, sae_id=sae_id_pre, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="SAE-based cluster classification for GPT2 DAS partition results"
    )
    parser.add_argument(
        "--partition-results-path",
        type=str,
        required=True,
        help="Path to partition_results_das JSON (e.g. partition_results_das_L7_P77_K2.json)",
    )
    parser.add_argument(
        "--graph-dataset-path",
        type=str,
        required=True,
        help="Path to graph_dataset pkl (e.g. graph_dataset_L7_P77_op5.pkl)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GPT2 model (default: util_model default)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        nargs="*",
        default=None,
        help="Layer(s) for activation extraction and SAE (e.g. --layer 5 7 9 or --layer \"0 1 2 3 4 5 6 7 8 9 10 11\"). Default: use layer from partition results",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=None,
        help="Token position for activation extraction (default: use pos_num from partition results)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for activation extraction",
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="SAE path or 'release:sae_id' (default: gpt2-small-res-jb for given layer)",
    )
    parser.add_argument(
        "--classifier-mode",
        type=str,
        choices=["top-k", "full-l1"],
        default="full-l1",
        help="top-k: use top 10 SAE features; full-l1: L1 logistic regression on all features",
    )
    parser.add_argument(
        "--full-l1-C",
        type=float,
        default=0.01,
        help="Inverse regularization strength for full-l1 mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: next to partition results)",
    )
    parser.add_argument(
        "--save-classifier",
        type=str,
        default=None,
        help="Path to save trained classifier (.pkl + .json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for feature firing in analysis",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Load partition results
    partition_path = Path(args.partition_results_path)
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition results not found: {partition_path}")
    with open(partition_path, "r") as f:
        partition_results = json.load(f)
    partition_layer = partition_results["layer"]
    partition_pos = partition_results["pos_num"]
    # SAE/activation layers: parse from args (supports "0 1 2" or "0,1,2" as single string)
    if args.layer is not None and len(args.layer) > 0:
        layers = []
        for part in args.layer:
            for token in part.replace(",", " ").split():
                layers.append(int(token))
    else:
        layers = [partition_layer]
    pos_num = args.pos if args.pos is not None else partition_pos
    labels = partition_results["labels"]
    iia_by_cluster = partition_results.get("iia_by_cluster", {})
    n_samples = partition_results.get("sample_size", len(labels))

    # Load graph dataset
    graph_path = Path(args.graph_dataset_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph dataset not found: {graph_path}")
    with open(graph_path, "rb") as f:
        graph_dataset = pickle.load(f)
    if len(graph_dataset) != n_samples and len(graph_dataset) != len(labels):
        raise ValueError(
            f"Graph dataset size {len(graph_dataset)} does not match partition labels {len(labels)} / sample_size {n_samples}"
        )
    labels = labels[: len(graph_dataset)]

    # Identify clusters (0 = high IIA, 1 = low IIA)
    cluster_0_id = 0
    cluster_1_id = 1
    if iia_by_cluster.get(str(cluster_1_id), 0) > iia_by_cluster.get(str(cluster_0_id), 0):
        cluster_0_id, cluster_1_id = 1, 0

    print("=" * 70)
    print("GPT2 DAS partition – SAE cluster classification")
    print("=" * 70)
    print(f"  Partition: {partition_path.name} (partition layer={partition_layer}, pos={partition_pos})")
    print(f"  SAE/activations: layers={layers}, pos={pos_num}")
    print(f"  Samples: {len(graph_dataset)}, labels: {len(labels)}")
    print(f"  IIA by cluster: {iia_by_cluster}")
    print(f"  Device: {device}")
    print()

    # Load model and tokenizer
    print("Loading GPT2 model and tokenizer...")
    model, tokenizer = util_model.load_model(args.model_path)
    model = model.to(device)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not SAE_LENS_AVAILABLE or SAE is None:
        print("ERROR: sae_lens not available. Install with: pip install sae-lens")
        return 1

    labels_arr = np.array(labels)
    cluster_mask = (labels_arr == cluster_0_id) | (labels_arr == cluster_1_id)
    y_all = (labels_arr[cluster_mask] == cluster_1_id).astype(int)

    results_per_layer = []
    classifiers_by_layer = {}

    for layer in layers:
        # Extract activations at (layer, pos_num)
        print(f"Extracting activations at layer={layer}, pos={pos_num}...")
        activations = extract_activations_at_layer_pos(
            model,
            tokenizer,
            graph_dataset,
            layer,
            pos_num,
            device,
            batch_size=args.batch_size,
        )
        activations = activations.to(device)
        # Load SAE for this layer
        print(f"Loading SAE for layer {layer}...")
        sae = load_sae_gpt2(layer, device=device, sae_path=args.sae_path)
        with torch.no_grad():
            sae_out = sae.encode(activations)
            if hasattr(sae_out, "feature_acts"):
                feat = sae_out.feature_acts
            elif isinstance(sae_out, torch.Tensor):
                feat = sae_out
            else:
                feat = sae_out[0]
        feature_activations = feat.cpu().numpy()
        del sae
        torch.cuda.empty_cache()

        # Train LogisticRegression on this layer's SAE features only
        X = feature_activations[cluster_mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=args.full_l1_C, max_iter=2000, random_state=42
        )
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        report = classification_report(
            y_test,
            clf.predict(X_test),
            target_names=[f"Cluster {cluster_0_id}", f"Cluster {cluster_1_id}"],
            output_dict=True,
            zero_division=0,
        )
        n_nonzero = int(np.count_nonzero(clf.coef_))

        layer_result = {
            "layer": layer,
            "n_features": int(X.shape[1]),
            "n_nonzero_features": n_nonzero,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "classification_report": report,
        }
        results_per_layer.append(layer_result)
        classifiers_by_layer[layer] = clf
        print(f"  Layer {layer}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, n_nonzero={n_nonzero}")

    model.cpu()
    torch.cuda.empty_cache()

    # Summary
    results = {
        "cluster_0_id": cluster_0_id,
        "cluster_1_id": cluster_1_id,
        "cluster_0_size": int((labels_arr == cluster_0_id).sum()),
        "cluster_1_size": int((labels_arr == cluster_1_id).sum()),
        "results_per_layer": results_per_layer,
        "layers": layers,
    }
    print("\nPer-layer classifier results:")
    print(f"  {'Layer':<8} {'Train':<8} {'Test':<8} {'n_feat':<8} {'n_nonzero':<10}")
    print("  " + "-" * 42)
    for r in results_per_layer:
        print(f"  {r['layer']:<8} {r['train_accuracy']:<8.4f} {r['test_accuracy']:<8.4f} {r['n_features']:<8} {r['n_nonzero_features']:<10}")
    best = max(results_per_layer, key=lambda x: x["test_accuracy"])
    print(f"  Best test accuracy: layer {best['layer']} ({best['test_accuracy']:.4f})")

    results["metadata"] = {
        "partition_results_path": str(partition_path),
        "graph_dataset_path": str(graph_path),
        "partition_layer": partition_layer,
        "partition_pos": partition_pos,
        "sae_layers": layers,
        "sae_pos": pos_num,
        "classifier_mode": args.classifier_mode,
        "sae_path": args.sae_path or "gpt2-small-res-jb",
        "best_layer_by_test_accuracy": best["layer"],
        "best_test_accuracy": best["test_accuracy"],
    }

    if args.save_classifier:
        save_path = Path(args.save_classifier)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifiers_by_layer, save_path.with_suffix(".pkl"))
        meta = {"layers": layers, "cluster_0_id": cluster_0_id, "cluster_1_id": cluster_1_id}
        with open(save_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Classifiers saved to {save_path.with_suffix('.pkl')} (dict: layer -> classifier)")

    layer_str = "_".join(map(str, layers))
    out_path = Path(args.output) if args.output else partition_path.parent / f"sae_classifier_L{layer_str}_P{pos_num}.json"
    if out_path.is_dir():
        out_path = out_path / f"sae_classifier_L{layer_str}_P{pos_num}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
