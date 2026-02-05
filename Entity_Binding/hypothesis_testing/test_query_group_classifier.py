#!/usr/bin/env python
"""
Filter Entity Binding Dataset Using Query Group Partitioning

This script:
1. Generates a dataset of entity binding samples (input only, no counterfactuals)
2. Uses query_group to partition samples into two clusters based on keep_indices
3. Filters out samples from one cluster (keeps cluster 0 with specified query_group indices)
4. Performs filtering like in partition_graph.py
5. Builds graph and calculates IIA for the filtered cluster
"""

import os
import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Add causalab and hypothesis_testing to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..', 'causalab'))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import functions from partition_graph.py
from partition_graph import (
    create_config,
    generate_input_samples,
    filter_input_samples,
    build_graph,
    build_directed_graph,
    compute_directed_and_undirected_density_from_directed,
    checker,
)

from causalab.tasks.entity_binding.config import (
    EntityBindingTaskConfig,
    create_sample_action_config,
    create_filling_liquids_config,
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
)
from causalab.neural.pipeline import LMPipeline

# Register Qwen3 support
def register_qwen3_for_pyvene():
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
        )
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        print("Successfully registered Qwen3 support for pyvene")
    except Exception as e:
        print(f"Warning: Could not register Qwen3: {e}")

register_qwen3_for_pyvene()


def predict_cluster_with_query_group(
    samples: List[Dict],
    num_groups: int,
    keep_indices: List[int]
) -> Tuple[np.ndarray, Dict]:
    """
    Use query_group to partition samples into two clusters.
    
    Args:
        samples: List of input samples (each should have 'query_group' field)
        num_groups: Number of groups (query_group values range from 0 to num_groups-1)
        keep_indices: List of query_group indices to keep in cluster 0 (others go to cluster 1)
        
    Returns:
        Tuple of (predictions array, metadata dict)
        - predictions: Array of cluster IDs (0 or 1) for each sample
        - metadata: Dict with cluster_0_id, cluster_1_id, and partition_info
    """
    # Validate keep_indices
    keep_indices_set = set(keep_indices)
    if not keep_indices_set.issubset(set(range(num_groups))):
        invalid = keep_indices_set - set(range(num_groups))
        raise ValueError(f"Invalid keep_indices: {invalid}. Must be in range [0, {num_groups-1}]")
    
    print(f"Partitioning samples by query_group:")
    print(f"  Cluster 0 (keep): query_groups {sorted(keep_indices_set)}")
    
    # Extract query_group for each sample
    query_groups = [sample.get('query_group', -1) for sample in samples]
    
    # Verify all samples have query_group
    if -1 in query_groups:
        raise ValueError("Some samples are missing 'query_group' field")
    
    # Determine cluster assignment: cluster 0 for keep_indices, cluster 1 for others
    cluster_0_query_groups = keep_indices_set
    cluster_1_query_groups = set(range(num_groups)) - keep_indices_set
    
    # Assign cluster based on query_group
    predictions = np.array([
        0 if qg in cluster_0_query_groups else 1
        for qg in query_groups
    ])
    
    # Create metadata
    partition_info = {
        "keep_indices": sorted(list(keep_indices_set)),
        "cluster_0_query_groups": sorted(list(cluster_0_query_groups)),
        "cluster_1_query_groups": sorted(list(cluster_1_query_groups)),
    }
    
    metadata = {
        "cluster_0_id": 0,
        "cluster_1_id": 1,
        "partition_info": partition_info,
        "num_groups": num_groups,
    }
    
    # Count distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nQuery group distribution:")
    print(f"  Cluster 0 (query_groups {partition_info['cluster_0_query_groups']}): {counts[0] if 0 in unique else 0} samples")
    print(f"  Cluster 1 (query_groups {partition_info['cluster_1_query_groups']}): {counts[1] if 1 in unique else 0} samples")
    
    return predictions, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Filter entity binding dataset using query_group partitioning"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="action",
        choices=["action", "filling_liquids"],
        help="Task type"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Initial sample size to generate"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=1,
        help="Query index"
    )
    parser.add_argument(
        "--answer-index",
        type=int,
        default=0,
        help="Answer index"
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="Number of groups"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./query_group_filter_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--keep-indices",
        type=int,
        nargs="+",
        required=True,
        help="List of query_group indices to keep in cluster 0 (e.g., --keep-indices 0 1 2)"
    )
    parser.add_argument(
        "--filtered-dataset-path",
        type=str,
        default=None,
        help="Path to filtered dataset .pkl from test_sae_classifier.py. If set, use this dataset instead of generating and filtering (ensures same test set)."
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Path to whole-graph .pkl (e.g. from test_sae_classifier.py or a previous run). If set, load this graph instead of building it. Must match the filtered dataset size."
    )

    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda:0"
    else:
        device = "cpu"
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "_")
    
    print("=" * 70)
    print("Filter Entity Binding Dataset Using Query Group Partitioning")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Number of groups: {args.num_groups}")
    print(f"  Keep indices: {args.keep_indices}")
    print(f"  Device: {device}")
    print()
    
    # Load model
    print(f"Loading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=5,
        device=device,
        max_length=200,
    )
    pipeline.tokenizer.padding_side = "left"
    print("  Model loaded")
    
    # Create config and causal model
    config = create_config(args.task, args.num_groups)
    causal_model = create_positional_entity_causal_model(config)

    # Step 1 & 2: Get filtered dataset (either load from test_sae_classifier or generate + filter)
    if args.filtered_dataset_path is not None:
        path = Path(args.filtered_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Filtered dataset not found: {path}")
        print(f"\nLoading filtered dataset from {path}...")
        with open(path, 'rb') as f:
            filtered_samples = pickle.load(f)
        if not isinstance(filtered_samples, list) or len(filtered_samples) == 0:
            raise ValueError("Loaded dataset must be a non-empty list of samples.")
        print(f"  Loaded {len(filtered_samples)} samples (same test set as test_sae_classifier)")
    else:
        # Generate dataset
        print(f"\nGenerating {args.sample_size} input samples...")
        samples = generate_input_samples(
            config, causal_model, args.sample_size,
            args.query_index, args.answer_index, args.num_groups
        )
        # Filter samples (like in partition_graph.py)
        print("\nFiltering samples (keeping only perfect model performance)...")
        filtered_samples = filter_input_samples(
            samples, pipeline, causal_model, args.batch_size
        )
        if len(filtered_samples) == 0:
            raise ValueError("No samples passed filtering! Model cannot perform correctly on any samples.")
        print(f"  Kept {len(filtered_samples)}/{len(samples)} samples after filtering")
    
    # Step 3: Partition samples by query_group
    print(f"\nPartitioning {len(filtered_samples)} samples by query_group...")
    predictions, partition_metadata = predict_cluster_with_query_group(
        filtered_samples,
        args.num_groups,
        args.keep_indices
    )
    
    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPartition distribution:")
    for cluster_id, count in zip(unique, counts):
        query_groups = partition_metadata["partition_info"][f"cluster_{cluster_id}_query_groups"]
        print(f"  Cluster {cluster_id} (query_groups {query_groups}): {count} samples")
    
    # Step 4: Summary by cluster (for reporting only)
    cluster_to_keep = 0
    cluster_to_filter = 1

    keep_mask = predictions == cluster_to_keep
    filtered_by_query_group = [s for s, keep in zip(filtered_samples, keep_mask) if keep]
    filtered_out_samples = [s for s, keep in zip(filtered_samples, keep_mask) if not keep]

    print(f"\nCluster {cluster_to_keep}: {len(filtered_by_query_group)} samples")
    print(f"Cluster {cluster_to_filter}: {len(filtered_out_samples)} samples")

    # Step 5: Get graph for the whole filtered dataset (build or load)
    full_adj_matrix = None
    full_iia = None
    full_undirected_density = None
    cluster_iia = None
    cluster_undirected_density = None
    filtered_out_iia = None
    filtered_out_undirected_density = None

    if len(filtered_samples) >= 2:
        if args.graph_path is not None:
            path = Path(args.graph_path)
            if not path.exists():
                raise FileNotFoundError(f"Graph file not found: {path}")
            print(f"\nLoading graph from {path}...")
            with open(path, 'rb') as f:
                full_adj_matrix = pickle.load(f)
            full_adj_matrix = np.asarray(full_adj_matrix)
            expected = (len(filtered_samples), len(filtered_samples))
            if full_adj_matrix.shape != expected:
                raise ValueError(
                    f"Loaded graph shape {full_adj_matrix.shape} does not match dataset size {len(filtered_samples)} (expected {expected})"
                )
            print(f"  Loaded graph with {full_adj_matrix.shape[0]} nodes")
            full_iia, full_undirected_density = compute_directed_and_undirected_density_from_directed(full_adj_matrix)
            print(f"  IIA for whole graph: {full_iia:.4f}")
            print(f"  Undirected density for whole graph: {full_undirected_density:.4f}")
        else:
            print(f"\nBuilding DIRECTED graph for whole filtered dataset ({len(filtered_samples)} samples)...")
            full_adj_matrix = build_directed_graph(
                pipeline, causal_model, config, filtered_samples,
                args.layer, args.batch_size
            )
            full_iia, full_undirected_density = compute_directed_and_undirected_density_from_directed(full_adj_matrix)
            print(f"  IIA for whole graph: {full_iia:.4f}")
            print(f"  Undirected density for whole graph: {full_undirected_density:.4f}")
    else:
        print(f"\nSkipping graph (only {len(filtered_samples)} samples, need at least 2)")

    # Compute IIA and density for each cluster from subgraphs of the whole graph (no extra model calls)
    if full_adj_matrix is not None:
        kept_idx = np.where(keep_mask)[0]
        filtered_out_idx = np.where(~keep_mask)[0]
        if len(kept_idx) >= 2:
            sub_kept = full_adj_matrix[np.ix_(kept_idx, kept_idx)]
            cluster_iia, cluster_undirected_density = compute_directed_and_undirected_density_from_directed(sub_kept)
            print(f"\n  IIA for kept cluster {cluster_to_keep} (from whole graph): {cluster_iia:.4f}")
            print(f"  Undirected density for kept cluster {cluster_to_keep}: {cluster_undirected_density:.4f}")
        if len(filtered_out_idx) >= 2:
            sub_filtered_out = full_adj_matrix[np.ix_(filtered_out_idx, filtered_out_idx)]
            filtered_out_iia, filtered_out_undirected_density = compute_directed_and_undirected_density_from_directed(sub_filtered_out)
            print(f"  IIA for filtered-out cluster {cluster_to_filter} (from whole graph): {filtered_out_iia:.4f}")
            print(f"  Undirected density for filtered-out cluster {cluster_to_filter}: {filtered_out_undirected_density:.4f}")

    print(f"\nResults:")
    print(f"  After model filtering: {len(filtered_samples)} samples")
    print(f"  Cluster {cluster_to_keep}: {len(filtered_by_query_group)} samples")
    print(f"  Cluster {cluster_to_filter}: {len(filtered_out_samples)} samples")
    if full_iia is not None:
        print(f"  IIA for whole graph: {full_iia:.4f}")
    if cluster_iia is not None:
        print(f"  IIA for kept cluster {cluster_to_keep}: {cluster_iia:.4f}")
    if filtered_out_iia is not None:
        print(f"  IIA for filtered-out cluster {cluster_to_filter}: {filtered_out_iia:.4f}")

    # Create filename suffix from keep_indices
    keep_indices_str = "_".join(map(str, sorted(args.keep_indices)))

    # Save results (include cluster_labels in JSON; one entry per sample in same order as filtered_samples)
    results = {
        "task": args.task,
        "num_groups": args.num_groups,
        "model": args.model,
        "layer": args.layer,
        "partition_metadata": partition_metadata,
        "initial_sample_size": args.sample_size if args.filtered_dataset_path is None else None,
        "filtered_dataset_path": args.filtered_dataset_path,
        "graph_path": args.graph_path,
        "after_model_filtering": len(filtered_samples),
        "cluster_kept": int(cluster_to_keep),
        "cluster_filtered": int(cluster_to_filter),
        "count_cluster_kept": len(filtered_by_query_group),
        "count_cluster_filtered": len(filtered_out_samples),
        "partition_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
        "iia_whole_graph": float(full_iia) if full_iia is not None else None,
        "undirected_density_whole_graph": float(full_undirected_density) if full_undirected_density is not None else None,
        "iia_kept_cluster": float(cluster_iia) if cluster_iia is not None else None,
        "undirected_density_kept_cluster": float(cluster_undirected_density) if cluster_undirected_density is not None else None,
        "iia_filtered_out_cluster": float(filtered_out_iia) if filtered_out_iia is not None else None,
        "undirected_density_filtered_out_cluster": float(filtered_out_undirected_density) if filtered_out_undirected_density is not None else None,
        "cluster_labels": [int(p) for p in predictions],
    }

    results_path = output_dir / f"query_group_filter_results_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save whole filtered dataset (one file)
    dataset_path = output_dir / f"filtered_dataset_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
    with open(dataset_path, 'wb') as f:
        pickle.dump(filtered_samples, f)

    # Save whole graph (one file)
    if full_adj_matrix is not None:
        graph_path = output_dir / f"graph_whole_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(full_adj_matrix, f)
        print(f"Graph (whole) saved to: {graph_path}")

    print(f"\nResults saved to: {results_path}")
    print(f"Filtered dataset (whole) saved to: {dataset_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
