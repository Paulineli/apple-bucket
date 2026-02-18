#!/usr/bin/env python
"""
DAS-based graph generation and partitioning (no causalab).

1. Generate graph_dataset similar to make_counterfactual_dataset_all + filter, keep only base. Save graph_dataset.
2. Build graph: nodes = samples in graph_dataset; edge (a,b) iff (a as source, b as base) and (b as source, a as base)
   are both predicted correctly by the given DAS. Save the graph.
3. Partition the graph using quasi-clique from partition_graph.py.
4. Reuses das.py and util_data.py (no causalab).
"""

import os
import sys
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# Add Entity_Binding/hypothesis_testing for partition_graph_quasi_clique
_script_dir = os.path.dirname(os.path.abspath(__file__))
_partition_dir = os.path.join(_script_dir, "..", "Entity_Binding", "hypothesis_testing")
if _partition_dir not in sys.path:
    sys.path.insert(0, _partition_dir)

import util_data
import util_model
import das
from partition_graph_quasi_clique import quasi_clique_partition


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_graph_dataset(
    data_generator: str,
    intervention: str,
    op_out: str,
    causal_model,
    vocab: List,
    texts: List,
    labels: List,
    model,
    tokenizer,
    data_size: int,
    device: str,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Generate graph_dataset: same as make_counterfactual_dataset (all + filter) but keep only
    the base part of each sample. Each element has: input_ids (dict), context_texts,
    context_labels, base_labels.
    """
    if data_generator == "all":
        make_raw = util_data.make_counterfactual_dataset_all
    elif data_generator == "all2":
        make_raw = util_data.make_counterfactual_dataset_all2
    else:
        raise ValueError("data_generator must be 'all' or 'all2'")

    raw = make_raw(causal_model, vocab, intervention, data_size)

    for dp in raw:
        indices = random.sample(range(len(texts)), 5)
        dp["context_texts"] = [texts[j] for j in indices]
        dp["context_labels"] = [labels[j] for j in indices]
        indices = random.sample(range(len(texts)), 5)
        dp["context_texts_source"] = [texts[j] for j in indices]
        dp["context_labels_source"] = [labels[j] for j in indices]

    filtered = util_data.data_filter(
        op_out, causal_model, model, tokenizer, raw, device, batch_size=batch_size
    )
    print(f"Filtered dataset size: {len(filtered)} (batch_size={batch_size})")

    # Keep only base
    graph_dataset = []
    for dp in filtered:
        graph_dataset.append({
            "input_ids": dp["input_ids"],
            "context_texts": dp["context_texts"],
            "context_labels": dp["context_labels"],
            "base_labels": dp["base_labels"],
        })
    return graph_dataset


def tokenize_pair(
    sample_base: Dict,
    sample_source: Dict,
    expected_label: str,
    tokenizer,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Build one tokenized example: base=sample_base, source=sample_source, label=expected_label."""
    base_text = util_data.format_input(
        sample_base["input_ids"],
        sample_base["context_texts"],
        sample_base["context_labels"],
    )
    source_text = util_data.format_input(
        sample_source["input_ids"],
        sample_source["context_texts"],
        sample_source["context_labels"],
    )
    return {
        "input_ids": tokenizer(base_text, return_tensors="pt")["input_ids"].to(device),
        "source_input_ids": tokenizer(source_text, return_tensors="pt")["input_ids"].to(device),
        "labels": tokenizer(str(expected_label), return_tensors="pt")["input_ids"].to(device),
    }


def _batch_tokenize_texts(
    tokenizer,
    texts: List[str],
    device: str,
    left_pad: bool = True,
) -> torch.Tensor:
    """Tokenize a list of texts in one batch. Returns (N, L) tensor, left-padded for causal LM."""
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


def build_graph(
    graph_dataset: List[Dict],
    causal_model,
    intervention: str,
    op_out: str,
    intervenable,
    pos: int,
    tokenizer,
    device: str,
    batch_size: int = 32,
    intervention_type: str = "das",
) -> np.ndarray:
    """
    Build undirected adjacency matrix. Edge (i,j) exists iff both
    (i as source, j as base) and (j as source, i as base) are predicted correctly by DAS.
    Uses batched DAS testing for efficiency.
    """
    n = len(graph_dataset)
    adj = np.zeros((n, n), dtype=bool)

    # Precompute base text string for each node (no tokenization yet)
    base_texts = []
    for i in tqdm(range(n), desc="Precomputing base texts", unit=" node"):
        base_texts.append(
            util_data.format_input(
                graph_dataset[i]["input_ids"],
                graph_dataset[i]["context_texts"],
                graph_dataset[i]["context_labels"],
            )
        )

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Collect (base_text, source_text, label_str) for each directed example; avoid per-pair tokenization
    flat_base_texts: List[str] = []
    flat_source_texts: List[str] = []
    flat_label_strs: List[str] = []
    pair_direction_index: List[Tuple[int, int, str]] = []  # (i, j, 'ij'|'ji') for each flat entry

    for i, j in tqdm(pairs, desc="Building pair examples", unit=" pair"):
        intervened_ij = {**graph_dataset[i]["input_ids"], intervention: graph_dataset[j]["base_labels"][intervention]}
        label_ij = str(causal_model.run_forward(intervened_ij)[op_out])
        intervened_ji = {**graph_dataset[j]["input_ids"], intervention: graph_dataset[i]["base_labels"][intervention]}
        label_ji = str(causal_model.run_forward(intervened_ji)[op_out])

        flat_base_texts.append(base_texts[i])
        flat_source_texts.append(base_texts[j])
        flat_label_strs.append(label_ij)
        pair_direction_index.append((i, j, "ij"))

        flat_base_texts.append(base_texts[j])
        flat_source_texts.append(base_texts[i])
        flat_label_strs.append(label_ji)
        pair_direction_index.append((i, j, "ji"))

    # Batch tokenize all texts (3 calls instead of 6 * len(pairs))
    num_examples = len(flat_base_texts)
    print(f"Batch tokenizing {num_examples} examples...")
    batch_input_ids = _batch_tokenize_texts(tokenizer, flat_base_texts, device, left_pad=True)
    batch_source_ids = _batch_tokenize_texts(tokenizer, flat_source_texts, device, left_pad=True)
    batch_labels = _batch_tokenize_texts(tokenizer, flat_label_strs, device, left_pad=True)

    # Build flat_examples as list of per-example dicts (sliced from batches) for downstream DAS batching
    flat_examples = [
        {
            "input_ids": batch_input_ids[i : i + 1],
            "source_input_ids": batch_source_ids[i : i + 1],
            "labels": batch_labels[i : i + 1],
        }
        for i in range(num_examples)
    ]

    print(f"Building graph: {len(pairs)} pairs, {num_examples} DAS checks in batches of {batch_size}...")

    all_correct = []
    num_batches = (len(flat_examples) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(flat_examples), batch_size), desc="DAS inference", total=num_batches, unit=" batch"):
        batch_list = flat_examples[start : start + batch_size]
        correct_batch = das.das_test_batch_correctness(
            intervenable, pos, batch_list, device, intervention_type=intervention_type
        )
        all_correct.extend(correct_batch)

    # Map back: for each pair (i,j) we have two entries, 'ij' and 'ji'; edge iff both correct
    pair_to_scores = {}  # (i,j) -> (correct_ij, correct_ji)
    for idx, (i, j, direction) in enumerate(pair_direction_index):
        key = (min(i, j), max(i, j))
        if key not in pair_to_scores:
            pair_to_scores[key] = {}
        pair_to_scores[key][direction] = all_correct[idx]

    for (i, j), scores in pair_to_scores.items():
        if scores.get("ij", False) and scores.get("ji", False):
            adj[i, j] = True
            adj[j, i] = True

    return adj


def compute_overall_iia(adj_matrix: np.ndarray) -> float:
    """Edge density for the full graph (undirected)."""
    n = adj_matrix.shape[0]
    if n <= 1:
        return 0.0
    num_edges = int(np.sum(adj_matrix)) // 2
    max_edges = n * (n - 1) // 2
    return num_edges / max_edges if max_edges > 0 else 0.0


def compute_subgraph_iia(adj_matrix: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    """Edge density for one cluster (undirected)."""
    mask = labels == cluster_id
    subgraph = adj_matrix[mask][:, mask]
    n = subgraph.shape[0]
    if n <= 1:
        return 0.0
    num_edges = int(np.sum(subgraph)) // 2
    max_edges = n * (n - 1) // 2
    return num_edges / max_edges if max_edges > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="DAS graph: generate graph_dataset, build graph, partition (no causalab)"
    )
    parser.add_argument("--causal-model", choices=["1", "2"], default="1")
    parser.add_argument("--data-size", type=int, default=200, help="Samples to generate before filter")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer", type=int, required=True, help="DAS layer")
    parser.add_argument("--pos-num", type=int, required=True, help="DAS position")
    parser.add_argument("--weights-path", type=str, default=None, help="DAS weights .pt")
    parser.add_argument("--candidates-path", type=str, default=None)
    parser.add_argument("--subspace-dimension", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-dataset-path", type=str, default=None, help="Load/save graph_dataset (base-only)")
    parser.add_argument("--graph-path", type=str, default=None, help="Load/save graph adjacency")
    parser.add_argument("--graph-batch-size", type=int, default=32, help="Batch size for DAS when building graph")
    parser.add_argument("--output-dir", type=str, default="./partition_results_das")
    parser.add_argument("--K", type=int, required=True, help="Number of partitions (quasi-cliques + remainder)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Min edge density for quasi-clique")
    parser.add_argument("--min-clique-size", type=int, default=2)
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Intervention to use (e.g. op5, op4a, op5a). If not set: op5 for causal-model 1, op4a for causal-model 2.",
    )

    args = parser.parse_args()

    set_random_seed(args.seed)

    device = args.device or (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab, texts, labels = util_data.get_vocab_and_data()
    if args.causal_model == "1":
        causal_model = util_data.build_causal_model(vocab)
        default_intervention = "op5"
        data_generator = "all"
        op_out = "op5"
    else:
        causal_model = util_data.build_causal_model2(vocab)
        default_intervention = "op4a"
        data_generator = "all2"
        op_out = "op6a"

    intervention = args.intervention if args.intervention is not None else default_intervention
    print(f"Using intervention: {intervention}")


    model, tokenizer = util_model.load_model()
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. graph_dataset -----
    if args.graph_dataset_path and Path(args.graph_dataset_path).exists():
        print(f"Loading graph_dataset from {args.graph_dataset_path}")
        with open(args.graph_dataset_path, "rb") as f:
            graph_dataset = pickle.load(f)
    else:
        print("Generating graph_dataset (base-only, filtered)...")
        graph_dataset = generate_graph_dataset(
            data_generator=data_generator,
            intervention=intervention,
            op_out=op_out,
            causal_model=causal_model,
            vocab=vocab,
            texts=texts,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            data_size=args.data_size,
            device=device,
            batch_size=args.batch_size,
        )
        if len(graph_dataset) == 0:
            raise ValueError("No samples in graph_dataset after filtering.")
        save_path = output_dir / f"graph_dataset_{args.data_size}_{intervention}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(graph_dataset, f)
        print(f"Saved graph_dataset to {save_path} (size={len(graph_dataset)})")
        if args.graph_dataset_path:
            with open(args.graph_dataset_path, "wb") as f:
                pickle.dump(graph_dataset, f)

    n = len(graph_dataset)

    # ----- 2. Graph -----
    if args.graph_path and Path(args.graph_path).exists():
        print(f"Loading graph from {args.graph_path}")
        with open(args.graph_path, "rb") as f:
            adj_matrix = pickle.load(f)
    else:
        weights_path = args.weights_path
        if not weights_path:
            weights_path = os.path.join(
                _script_dir,
                "training_results",
                f"das_weights_das_or_model_{args.causal_model}_dim{args.subspace_dimension}.pt",
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"DAS weights not found: {weights_path}")

        das_weights = das.load_weight(weights_path)
        intervention_weights = das_weights.get(intervention)
        if intervention_weights is None:
            intervention_weights = next(iter(das_weights.values()), {}) if das_weights else {}
        key = f"L{args.layer}_P{args.pos_num}"
        weight = intervention_weights.get(key) if isinstance(intervention_weights, dict) else None
        if weight is None:
            raise KeyError(f"Could not find DAS weight for layer={args.layer}, pos={args.pos_num} (key={key}) in {weights_path}")

        intervenable = das.config_das(
            model, args.layer, device,
            weight=weight,
            subspace_dimension=args.subspace_dimension,
        )

        print("Building graph from DAS predictions...")
        adj_matrix = build_graph(
            graph_dataset,
            causal_model,
            intervention,
            op_out,
            intervenable,
            args.pos_num,
            tokenizer,
            device,
            batch_size=args.graph_batch_size,
            intervention_type="das",
        )
        graph_save = output_dir / f"graph_das_L{args.layer}_P{args.pos_num}_{args.data_size}_{intervention}.pkl"
        with open(graph_save, "wb") as f:
            pickle.dump(adj_matrix, f)
        print(f"Saved graph to {graph_save} (nodes={n}, edges={int(np.sum(adj_matrix))//2})")
        if args.graph_path:
            with open(args.graph_path, "wb") as f:
                pickle.dump(adj_matrix, f)

    # ----- 3. Partition -----
    print(f"Partitioning graph (K={args.K}, gamma={args.gamma})...")
    labels_arr = quasi_clique_partition(
        adj_matrix, args.K, args.gamma, args.min_clique_size
    )

    overall_iia = compute_overall_iia(adj_matrix)
    print(f"\nOverall IIA (edge density): {overall_iia:.3f}")

    print("\nSubgraph IIA (edge density):")
    unique_labels = np.unique(labels_arr)
    iia_by_cluster = {}
    for k in unique_labels:
        k = int(k)
        iia = compute_subgraph_iia(adj_matrix, labels_arr, k)
        iia_by_cluster[k] = iia
        size = int(np.sum(labels_arr == k))
        print(f"  Cluster {k}: IIA = {iia:.3f} (size: {size})")

    results = {
        "causal_model": args.causal_model,
        "layer": args.layer,
        "pos_num": args.pos_num,
        "sample_size": n,
        "method": "quasi_clique",
        "K": args.K,
        "gamma": args.gamma,
        "min_clique_size": args.min_clique_size,
        "overall_iia": overall_iia,
        "iia_by_cluster": iia_by_cluster,
        "intervention": intervention,
        "labels": labels_arr.tolist(),
    }
    results_path = output_dir / f"partition_results_das_L{args.layer}_P{args.pos_num}_K{args.K}_{args.data_size}_{intervention}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    model.cpu()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
