#!/usr/bin/env python3
"""
Step 4: SAE-based classifier for factual recall graph-partition groups.

1. Load test_dataset.pkl + graph.pkl from artifacts/test_results
2. Partition graph into K clusters via Spectral Clustering; report IIA per cluster
3. Extract residual stream at layer 14, city last-token position (for the N graph nodes)
4. Encode through SAE → sparse feature activations
5. Train multi-class logistic regression predicting cluster label
6. Report SAE-based prediction accuracy; find most differential SAE features between clusters
"""

import sys, os, json, pickle, argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# causalab lives under Entity_Binding/causalab/causalab
sys.path.insert(0, str(_REPO_ROOT / "Entity_Binding" / "causalab"))
# hypothesis_testing scripts (quasi_clique_partition)
_HYP_DIR = _REPO_ROOT / "Entity_Binding" / "hypothesis_testing"
if str(_HYP_DIR) not in sys.path:
    sys.path.insert(0, str(_HYP_DIR))

from partition_graph_quasi_clique import quasi_clique_partition

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# Load HF token if present
_hf_token_path = Path(__file__).parent.parent / "hf_token.txt"
if _hf_token_path.exists():
    for line in _hf_token_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            os.environ["HF_TOKEN"] = line
            break

MODEL_NAME = "meta-llama/Llama-3.1-8B"
_HF_CACHE = Path(__file__).resolve().parent.parent.parent / ".hf_cache"
_LOCAL_MODEL = _HF_CACHE / "models--meta-llama--Llama-3.1-8B" / "snapshots" / "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MODEL_PATH = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else MODEL_NAME


# ---------------------------------------------------------------------------
# Model loading (returns model + tokenizer directly; no LMPipeline needed)
# ---------------------------------------------------------------------------

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


# ---------------------------------------------------------------------------
# City last-token finder
# ---------------------------------------------------------------------------

def find_city_last_token(tokenizer, prompt: str, city: str) -> int:
    """Return the index of the last token of `city` in the tokenized `prompt`."""
    full_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    # Try city with and without a leading space (Llama tokenizer adds space prefix)
    for prefix in ["", " ", "\u2581"]:
        city_ids = tokenizer(prefix + city, add_special_tokens=False).input_ids
        if not city_ids:
            continue
        n = len(city_ids)
        for end in range(len(full_ids) - 1, n - 2, -1):
            if full_ids[end - n + 1 : end + 1] == city_ids:
                return end
    return len(full_ids) - 1  # fallback: last token


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, samples: List[dict], layer: int) -> torch.Tensor:
    """Extract residual stream (block output) at `layer`, city last-token position."""
    device = next(model.parameters()).device
    acts = []

    for s in tqdm(samples, desc=f"Extracting layer-{layer} activations"):
        enc = tokenizer(s["prompt"], return_tensors="pt").to(device)
        city_idx = find_city_last_token(tokenizer, s["prompt"], s["city"])

        captured = [None]

        def _hook(module, inp, out):
            captured[0] = (out[0] if isinstance(out, tuple) else out).detach().float()

        h = model.model.layers[layer].register_forward_hook(_hook)
        with torch.no_grad():
            model(**enc)
        h.remove()

        acts.append(captured[0][0, city_idx, :].cpu())

    return torch.stack(acts)  # (N, d_model)


# ---------------------------------------------------------------------------
# SAE loading + encoding
# ---------------------------------------------------------------------------

def load_sae(sae_path: str, device: str):
    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError("sae_lens not installed. Run: pip install sae-lens")

    if os.path.exists(sae_path):
        return SAE.load_from_disk(sae_path, device=device)
    if ":" in sae_path:
        release, sae_id = sae_path.split(":", 1)
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        except TypeError:
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        return sae
    raise ValueError(f"sae_path must be 'release:sae_id' or a local path; got: {sae_path}")


def encode_sae(sae, activations: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    device = next(sae.parameters()).device
    chunks = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i : i + batch_size].to(device)
        with torch.no_grad():
            out = sae.encode(batch)
        if hasattr(out, "feature_acts"):
            fa = out.feature_acts
        elif isinstance(out, torch.Tensor):
            fa = out
        elif isinstance(out, tuple):
            fa = out[0]
        else:
            raise ValueError(f"Unexpected SAE output type: {type(out)}")
        chunks.append(fa.cpu())
    return torch.cat(chunks, dim=0)  # (N, n_sae_features)


# ---------------------------------------------------------------------------
# Graph partitioning + IIA
# (quasi_clique_partition imported from partition_graph_quasi_clique.py)
# ---------------------------------------------------------------------------


def compute_subgraph_iia(adj_matrix: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    """Edge density (IIA) of the subgraph induced by cluster_id."""
    mask = labels == cluster_id
    sub = adj_matrix[mask][:, mask]
    n = sub.shape[0]
    if n <= 1:
        return 0.0
    num_edges = np.sum(sub) // 2
    max_edges = n * (n - 1) // 2
    return float(num_edges / max_edges) if max_edges > 0 else 0.0


# ---------------------------------------------------------------------------
# Differential SAE features between clusters
# ---------------------------------------------------------------------------

def find_differential_features(X: np.ndarray, labels: np.ndarray, top_k: int = 20) -> Dict:
    """For each cluster find SAE features that fire most differently vs other clusters.
    Returns features with the largest (mean_in_this_cluster - mean_in_others) per cluster."""
    results = {}
    for lbl in sorted(set(labels.tolist())):
        mask_this = labels == lbl
        mask_other = ~mask_this
        mean_this = X[mask_this].mean(axis=0)
        mean_other = X[mask_other].mean(axis=0)
        # Differential: positive = fires more in this cluster than others
        diff = mean_this - mean_other
        top_idx = np.argsort(diff)[::-1][:top_k]
        firing_this = (X[mask_this] > 0).mean(axis=0)
        firing_other = (X[mask_other] > 0).mean(axis=0)
        results[int(lbl)] = {
            "n_samples": int(mask_this.sum()),
            "top_differential_features": top_idx.tolist(),
            "differential_mean": diff[top_idx].tolist(),
            "mean_in_cluster": mean_this[top_idx].tolist(),
            "mean_in_other_clusters": mean_other[top_idx].tolist(),
            "firing_rate_in_cluster": firing_this[top_idx].tolist(),
            "firing_rate_in_other": firing_other[top_idx].tolist(),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def directed_to_undirected(adj_directed: np.ndarray) -> np.ndarray:
    """Recover undirected graph from directed: edge (i,j) exists iff both i->j and j->i."""
    adj = np.asarray(adj_directed, dtype=bool)
    return adj & adj.T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-results-dir", type=str, default=None,
                   help="Test results dir (e.g. test_results_mdas); default: test_results")
    p.add_argument("--K", type=int, default=2,
                   help="Number of subgraphs for quasi-clique partitioning")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Minimum edge density for quasi-cliques (default: 0.9)")
    p.add_argument("--min-clique-size", type=int, default=2,
                   help="Minimum quasi-clique size (default: 2)")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--sae-path", type=str, required=True,
                   help="SAE: 'release:sae_id' or local path")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for SAE encoding")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--top-k-features", type=int, default=20,
                   help="Top differential SAE features to report per cluster")
    p.add_argument("--cache-activations", type=str, default=None,
                   help="Path to cache/load pre-extracted activations (.pt)")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir) if args.output_dir else ARTIFACTS / "classifier_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph and test dataset
    # ------------------------------------------------------------------
    test_results_dir = Path(args.test_results_dir) if args.test_results_dir else ARTIFACTS / "test_results"
    with open(test_results_dir / "test_dataset.pkl", "rb") as f:
        all_samples = pickle.load(f)
    with open(test_results_dir / "graph.pkl", "rb") as f:
        adj_directed = pickle.load(f)

    adj_directed = np.asarray(adj_directed, dtype=bool)
    n_nodes = adj_directed.shape[0]

    # Recover undirected graph for quasi-clique: edge (i,j) iff both i->j and j->i
    adj_matrix = directed_to_undirected(adj_directed)
    print(f"Graph: directed ({int(np.sum(adj_directed))} edges) -> undirected ({int(np.sum(adj_matrix))//2} edges)")

    # graph.pkl covers the first n_nodes samples of test_dataset
    samples = all_samples[:n_nodes]
    print(f"Graph nodes: {n_nodes}  |  test_dataset total: {len(all_samples)}")

    # ------------------------------------------------------------------
    # Partition graph → cluster labels (quasi-clique, same as partition_graph.py)
    # ------------------------------------------------------------------
    print(f"\nQuasi-clique partitioning into K={args.K} subgraphs (gamma={args.gamma})...")
    cluster_labels = quasi_clique_partition(
        adj_matrix, args.K, args.gamma, args.min_clique_size
    )

    # Overall IIA
    num_edges = int(np.sum(adj_matrix)) // 2
    max_edges = n_nodes * (n_nodes - 1) // 2
    overall_iia = num_edges / max_edges if max_edges > 0 else 0.0
    print(f"Overall graph IIA (edge density): {overall_iia:.4f}")

    print("\nCluster sizes and IIA (edge density within cluster):")
    cluster_iia = {}
    for k in range(args.K):
        iia = compute_subgraph_iia(adj_matrix, cluster_labels, k)
        size = int((cluster_labels == k).sum())
        cluster_iia[k] = iia
        print(f"  Cluster {k}: n={size:3d}, IIA={iia:.4f}")

    # ------------------------------------------------------------------
    # Extract activations (or load from cache)
    # ------------------------------------------------------------------
    cache_path = Path(args.cache_activations) if args.cache_activations else \
        output_dir / f"activations_layer{args.layer}_n{n_nodes}.pt"

    if cache_path.exists():
        print(f"\nLoading cached activations from {cache_path}")
        activations = torch.load(cache_path, map_location="cpu")
        if activations.shape[0] != n_nodes:
            print(f"  Cache size mismatch ({activations.shape[0]} vs {n_nodes}); re-extracting.")
            activations = None
        else:
            print(f"  Activations shape: {activations.shape}")
    else:
        activations = None

    if activations is None:
        print(f"\nLoading model {MODEL_NAME}...")
        model, tokenizer = load_model()
        print("  Model loaded")
        activations = extract_activations(model, tokenizer, samples, layer=args.layer)
        torch.save(activations, cache_path)
        print(f"  Saved activations to {cache_path}")
        del model, tokenizer
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Encode through SAE
    # ------------------------------------------------------------------
    print(f"\nLoading SAE from {args.sae_path}...")
    sae = load_sae(args.sae_path, device=device)
    print("  SAE loaded")

    print("Encoding activations through SAE...")
    feat_acts = encode_sae(sae, activations, batch_size=args.batch_size)
    print(f"  SAE features shape: {feat_acts.shape}")

    # ------------------------------------------------------------------
    # Train multi-class classifier (predict cluster from SAE features)
    # ------------------------------------------------------------------
    X = feat_acts.numpy()
    y = cluster_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}, Clusters: {args.K}")

    print("Training multi-class L1 logistic regression on SAE features...")
    clf = LogisticRegression(
        penalty="l1", solver="saga", C=0.1,
        max_iter=2000, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(
        y_test, clf.predict(X_test),
        target_names=[f"Cluster {k}" for k in range(args.K)],
        output_dict=True,
    )

    print(f"\n{'='*60}")
    print(f"SAE-based prediction accuracy")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(
        y_test, clf.predict(y_test if False else X_test),
        target_names=[f"Cluster {k}" for k in range(args.K)],
    ))

    # ------------------------------------------------------------------
    # L1 logistic regression selected features (non-zero coefficients)
    # ------------------------------------------------------------------
    coef = clf.coef_  # (n_classes, n_features) for multi-class
    n_total = coef.shape[1]
    l1_selected = {}
    for k in range(args.K):
        nonzero = np.where(coef[k] != 0)[0]
        # Sort by absolute coefficient (most influential first)
        order = np.argsort(np.abs(coef[k][nonzero]))[::-1]
        sorted_idx = nonzero[order]
        l1_selected[k] = {
            "feature_indices": sorted_idx.tolist(),
            "coefficients": coef[k][sorted_idx].tolist(),
        }
    all_selected = np.unique(np.concatenate([np.where(coef[k] != 0)[0] for k in range(args.K)]))
    print(f"\nL1 logistic regression selected features:")
    print(f"  Total SAE features: {n_total}, Non-zero in at least one class: {len(all_selected)}")
    for k in range(args.K):
        info = l1_selected[k]
        feats = info["feature_indices"][:10]
        coefs = [f"{c:.3f}" for c in info["coefficients"][:10]]
        print(f"  Cluster {k}: {len(info['feature_indices'])} selected  "
              f"top 10 features {feats}  coefs {coefs}")

    # ------------------------------------------------------------------
    # Differential SAE features between clusters
    # ------------------------------------------------------------------
    print(f"Finding top-{args.top_k_features} most differential SAE features per cluster...")
    differential_features = find_differential_features(X, y, top_k=args.top_k_features)

    print("\nMost differential SAE features per cluster (highest in this cluster vs others):")
    for k in range(args.K):
        info = differential_features[k]
        feats = info["top_differential_features"][:5]
        diffs = [f"{v:.3f}" for v in info["differential_mean"][:5]]
        in_cl = [f"{v:.3f}" for v in info["mean_in_cluster"][:5]]
        out_cl = [f"{v:.3f}" for v in info["mean_in_other_clusters"][:5]]
        print(f"  Cluster {k} (n={info['n_samples']:3d}, IIA={cluster_iia[k]:.3f}): "
              f"features {feats}  diff(mean) {diffs}  in_cluster {in_cl}  in_other {out_cl}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "layer": args.layer,
        "sae_path": args.sae_path,
        "K": args.K,
        "gamma": args.gamma,
        "n_graph_nodes": n_nodes,
        "overall_iia": float(overall_iia),
        "cluster_iia": {k: float(v) for k, v in cluster_iia.items()},
        "cluster_sizes": {k: int((cluster_labels == k).sum()) for k in range(args.K)},
        "cluster_labels": cluster_labels.tolist(),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "l1_selected_features_per_cluster": l1_selected,
        "l1_selected_count": int(len(all_selected)),
        "differential_features_per_cluster": differential_features,
    }

    out_path = output_dir / f"sae_classifier_results_K{args.K}_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    clf_path = output_dir / f"sae_classifier_K{args.K}_layer{args.layer}.pkl"
    joblib.dump({"classifier": clf, "K": args.K, "cluster_labels": cluster_labels}, clf_path)

    print(f"\nResults → {out_path}")
    print(f"Classifier → {clf_path}")


if __name__ == "__main__":
    main()
