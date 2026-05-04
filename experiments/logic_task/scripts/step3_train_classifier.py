#!/usr/bin/env python3
"""
Train two linear classifiers on GPT2 DAS partition data:

1) Natural features: use base labels for op1/op2/op3 as binary features.
2) SAE features: extract GPT2 residual-stream activations at (layer, pos), encode
   with a pretrained GPT2 residual SAE (via sae_lens), then fit an L1-regularized
   logistic regression classifier on SAE feature activations.

Saves both classifiers and reports top-5 coefficients.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Reuse the exact activation extraction + SAE loading from this repo.
from step4_analyze import (
    extract_activations_at_layer_pos,
    load_sae_gpt2,
)

import util_data
import util_model


def _bool_to_float01(x: bool | int | np.bool_ | np.integer) -> float:
    # Ensure we always get a plain Python float.
    return float(bool(x))


def build_natural_feature_matrix(graph_dataset: List[Dict[str, Any]]) -> np.ndarray:
    """
    Natural features are the truth values for op1/op2/op3 in base_labels.
    Returns X shape: (n_samples, 3).
    """
    feats = []
    for dp in graph_dataset:
        base_labels = dp["base_labels"]
        feats.append(
            [
                _bool_to_float01(base_labels["op1"]),
                _bool_to_float01(base_labels["op2"]),
                _bool_to_float01(base_labels["op3"]),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def infer_y_from_partition_labels(
    partition_results: Dict[str, Any],
    labels: List[int],
) -> Tuple[np.ndarray, int, int]:
    """
    partition_results["iia_by_cluster"] is a float per cluster id (0/1).
    We follow step4_analyze.py labeling convention:
      - cluster_0_id becomes the higher IIA cluster
      - cluster_1_id becomes the lower IIA cluster
      - y=1 means "lower IIA" membership
    """
    iia_by_cluster = partition_results.get("iia_by_cluster", {})
    cluster_0_id, cluster_1_id = 0, 1
    if iia_by_cluster.get(str(cluster_1_id), 0.0) > iia_by_cluster.get(str(cluster_0_id), 0.0):
        cluster_0_id, cluster_1_id = 1, 0

    labels_arr = np.asarray(labels, dtype=int)
    y = (labels_arr == cluster_1_id).astype(int)
    return y, cluster_0_id, cluster_1_id


def top_k_coefficients(
    coef_1d: np.ndarray,
    feature_names: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    coef_1d shape: (n_features,)
    Returns list sorted by absolute coefficient magnitude descending.
    """
    coef_1d = np.asarray(coef_1d).reshape(-1)
    k = min(k, coef_1d.shape[0])
    idx = np.argsort(np.abs(coef_1d))[::-1][:k]
    out = []
    for i in idx:
        out.append(
            {
                "feature": feature_names[i],
                "feature_index": int(i),
                "coefficient": float(coef_1d[i]),
                "abs_coefficient": float(abs(coef_1d[i])),
            }
        )
    return out


def save_classifier_with_metadata(
    clf: LogisticRegression,
    save_path_no_suffix: Path,
    metadata: Dict[str, Any],
) -> None:
    save_path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    clf_path = save_path_no_suffix.with_suffix(".pkl")
    meta_path = save_path_no_suffix.with_suffix(".json")
    joblib.dump(clf, clf_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train natural-op features and SAE L1 classifier.")
    parser.add_argument(
        "--partition-path",
        type=str,
        required=True,
        help="Path to partition_results_das JSON (e.g. partition_results_das_L5_P78_K2_200_op4.json).",
    )
    parser.add_argument(
        "--graph-dataset-path",
        type=str,
        required=True,
        help="Path to graph_dataset pickle aligned with partition labels.",
    )
    parser.add_argument(
        "--sae-C",
        type=float,
        default=0.01,
        help="L1 logistic regression C (smaller = stronger regularization).",
    )
    parser.add_argument(
        "--sae-max-iter",
        type=int,
        default=2000,
        help="Max iterations for L1 logistic regression.",
    )
    parser.add_argument(
        "--natural-C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for natural-feature logistic regression (L2).",
    )
    parser.add_argument(
        "--natural-max-iter",
        type=int,
        default=2000,
        help="Max iterations for natural-feature logistic regression.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: next to partition JSON).",
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="SAE path or 'release:sae_id' (default uses sae_lens gpt2-small-res-jb).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (if available).",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=None,
        help="Override transformer block index for SAE activations (default: layer from partition JSON).",
    )
    parser.add_argument(
        "--sae-pos",
        type=int,
        default=None,
        help="Override token position for SAE activations (default: pos_num from partition JSON).",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    partition_path = Path(args.partition_path)
    graph_dataset_path = Path(args.graph_dataset_path)
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition results not found: {partition_path}")
    if not graph_dataset_path.exists():
        raise FileNotFoundError(f"Graph dataset not found: {graph_dataset_path}")

    with open(partition_path, "r") as f:
        partition_results = json.load(f)

    partition_layer = int(partition_results["layer"])
    partition_pos = int(partition_results["pos_num"])
    sae_layer = int(args.sae_layer) if args.sae_layer is not None else partition_layer
    sae_pos = int(args.sae_pos) if args.sae_pos is not None else partition_pos
    intervention_name = str(partition_results.get("intervention", ""))

    with open(graph_dataset_path, "rb") as f:
        graph_dataset = pickle.load(f)

    labels = partition_results["labels"]
    # Align label length to dataset length (mirrors step4_analyze behavior).
    n_samples = int(partition_results.get("sample_size", len(labels)))
    if len(graph_dataset) != n_samples and len(graph_dataset) != len(labels):
        raise ValueError(
            f"Graph dataset size {len(graph_dataset)} does not match partition labels {len(labels)} / "
            f"sample_size {n_samples}. Ensure you pass the aligned graph_dataset pickle."
        )
    labels = labels[: len(graph_dataset)]

    y, cluster_0_id, cluster_1_id = infer_y_from_partition_labels(partition_results, labels)

    if args.output_dir is None:
        out_dir = partition_path.parent
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Natural-feature LR
    # -----------------
    X_nat = build_natural_feature_matrix(graph_dataset)
    feature_names_nat = ["op1", "op2", "op3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_nat, y, test_size=0.2, random_state=42, stratify=y
    )
    natural_clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=float(args.natural_C),
        max_iter=int(args.natural_max_iter),
        random_state=42,
    )
    natural_clf.fit(X_train, y_train)

    nat_train_acc = accuracy_score(y_train, natural_clf.predict(X_train))
    nat_test_acc = accuracy_score(y_test, natural_clf.predict(X_test))
    nat_top5 = top_k_coefficients(natural_clf.coef_.reshape(-1), feature_names_nat, k=5)

    natural_save_base = (
        out_dir
        / f"natural_lr_op1_op2_op3_L{partition_layer}_P{partition_pos}_intervention{intervention_name}"
    )
    save_classifier_with_metadata(
        clf=natural_clf,
        save_path_no_suffix=natural_save_base,
        metadata={
            "feature_type": "natural_op_features",
            "features": feature_names_nat,
            "partition_path": str(partition_path),
            "graph_dataset_path": str(graph_dataset_path),
            "layer": partition_layer,
            "pos_num": partition_pos,
            "cluster_0_id_high_iia": cluster_0_id,
            "cluster_1_id_low_iia": cluster_1_id,
            "label_definition": "y=1 iff sample is in lower IIA cluster",
            "train_accuracy": float(nat_train_acc),
            "test_accuracy": float(nat_test_acc),
            "classifier_params": {
                "penalty": "l2",
                "solver": "liblinear",
                "C": float(args.natural_C),
                "max_iter": int(args.natural_max_iter),
                "random_state": 42,
            },
            "top_coefficients": nat_top5,
        },
    )

    # -----------------
    # SAE-feature LR (L1)
    # -----------------
    model, tokenizer = util_model.load_model(None)
    model = model.to(device)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract residual-stream activations at (sae_layer, sae_pos); may differ from DAS partition (layer, pos).
    activations = extract_activations_at_layer_pos(
        model=model,
        tokenizer=tokenizer,
        graph_dataset=graph_dataset,
        layer=sae_layer,
        pos_num=sae_pos,
        device=device,
        batch_size=32,
    )
    activations = activations.to(device)

    sae = load_sae_gpt2(sae_layer, device=device, sae_path=args.sae_path)
    with torch.no_grad():
        sae_out = sae.encode(activations)
        if hasattr(sae_out, "feature_acts"):
            feature_activations = sae_out.feature_acts
        elif isinstance(sae_out, torch.Tensor):
            feature_activations = sae_out
        else:
            feature_activations = sae_out[0]
    feature_activations_np = feature_activations.detach().cpu().numpy()
    del sae, model, activations
    torch.cuda.empty_cache()

    X_sae = feature_activations_np
    n_features_sae = X_sae.shape[1]
    feature_names_sae = [f"sae_f{idx}" for idx in range(n_features_sae)]

    X_train_sae, X_test_sae, y_train_sae, y_test_sae = train_test_split(
        X_sae, y, test_size=0.2, random_state=42, stratify=y
    )
    sae_clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=float(args.sae_C),
        max_iter=int(args.sae_max_iter),
        random_state=42,
        n_jobs=-1,
    )
    sae_clf.fit(X_train_sae, y_train_sae)

    sae_train_acc = accuracy_score(y_train_sae, sae_clf.predict(X_train_sae))
    sae_test_acc = accuracy_score(y_test_sae, sae_clf.predict(X_test_sae))
    sae_top5 = top_k_coefficients(sae_clf.coef_.reshape(-1), feature_names_sae, k=5)

    sae_save_base = (
        out_dir
        / f"sae_l1_lr_full_features_L{sae_layer}_P{sae_pos}_intervention{intervention_name}"
    )
    save_classifier_with_metadata(
        clf=sae_clf,
        save_path_no_suffix=sae_save_base,
        metadata={
            "feature_type": "sae_feature_activations",
            "partition_layer": partition_layer,
            "partition_pos_num": partition_pos,
            "sae_layer": sae_layer,
            "sae_pos_num": sae_pos,
            "partition_path": str(partition_path),
            "graph_dataset_path": str(graph_dataset_path),
            "cluster_0_id_high_iia": cluster_0_id,
            "cluster_1_id_low_iia": cluster_1_id,
            "label_definition": "y=1 iff sample is in lower IIA cluster",
            "train_accuracy": float(sae_train_acc),
            "test_accuracy": float(sae_test_acc),
            "classifier_params": {
                "penalty": "l1",
                "solver": "saga",
                "C": float(args.sae_C),
                "max_iter": int(args.sae_max_iter),
                "random_state": 42,
                "n_jobs": -1,
            },
            "n_features": int(n_features_sae),
            "n_nonzero_coefficients": int(np.count_nonzero(sae_clf.coef_)),
            "top_coefficients": sae_top5,
        },
    )

    # Save a compact summary with both top-5 lists.
    if args.sae_layer is not None or args.sae_pos is not None:
        summary_path = (
            out_dir
            / f"linear_classifier_coeff_summary_L{partition_layer}_P{partition_pos}_saeL{sae_layer}_P{sae_pos}.json"
        )
    else:
        summary_path = (
            out_dir
            / f"linear_classifier_coeff_summary_L{partition_layer}_P{partition_pos}.json"
        )
    with open(summary_path, "w") as f:
        json.dump(
            {
                "partition_path": str(partition_path),
                "graph_dataset_path": str(graph_dataset_path),
                "partition_layer": partition_layer,
                "partition_pos_num": partition_pos,
                "sae_layer": sae_layer,
                "sae_pos_num": sae_pos,
                "cluster_0_id_high_iia": cluster_0_id,
                "cluster_1_id_low_iia": cluster_1_id,
                "natural_classifier": {
                    "model_path": str(natural_save_base.with_suffix(".pkl")),
                    "test_accuracy": float(nat_test_acc),
                    "top5": nat_top5,
                },
                "sae_classifier": {
                    "model_path": str(sae_save_base.with_suffix(".pkl")),
                    "test_accuracy": float(sae_test_acc),
                    "top5": sae_top5,
                },
            },
            f,
            indent=2,
        )

    # Print results (requested).
    print("\nTop 5 coefficients: Natural features logistic regression (op1/op2/op3)")
    for row in nat_top5:
        print(f"  {row['feature']}: coef={row['coefficient']:.6f}  abs={row['abs_coefficient']:.6f}")
    print(f"  Train acc={nat_train_acc:.4f} Test acc={nat_test_acc:.4f}")

    print("\nTop 5 coefficients: SAE-feature L1 logistic regression")
    for row in sae_top5:
        print(f"  {row['feature']}: coef={row['coefficient']:.6f}  abs={row['abs_coefficient']:.6f}")
    print(f"  Train acc={sae_train_acc:.4f} Test acc={sae_test_acc:.4f}")
    print(f"\nSaved models + metadata under: {out_dir}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

