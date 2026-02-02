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
    compute_subgraph_iia,
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
    
    # Step 1: Generate dataset
    print(f"\nGenerating {args.sample_size} input samples...")
    samples = generate_input_samples(
        config, causal_model, args.sample_size,
        args.query_index, args.answer_index, args.num_groups
    )
    
    # Step 2: Filter samples (like in partition_graph.py)
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
    
    # Step 4: Filter to separate clusters (always keep cluster 0, which contains keep_indices)
    cluster_to_keep = 0
    cluster_to_filter = 1
    
    keep_mask = predictions == cluster_to_keep
    filtered_by_query_group = [s for s, keep in zip(filtered_samples, keep_mask) if keep]
    filtered_out_samples = [s for s, keep in zip(filtered_samples, keep_mask) if not keep]
    
    print(f"\nFiltered to {len(filtered_by_query_group)} samples in cluster {cluster_to_keep}")
    print(f"Filtered out {len(filtered_out_samples)} samples in cluster {cluster_to_filter}")
    
    # Step 5: Build graphs and compute IIA for both clusters
    cluster_iia = None
    filtered_out_iia = None
    
    # Build graph for kept cluster
    if len(filtered_by_query_group) >= 2:
        print(f"\nBuilding DIRECTED graph for kept cluster (cluster {cluster_to_keep})...")
        adj_matrix_kept = build_directed_graph(
            pipeline, causal_model, config, filtered_by_query_group,
            args.layer, args.batch_size
        )
        
        # Compute IIA for kept cluster
        labels_kept = np.zeros(len(filtered_by_query_group), dtype=int)
        cluster_iia = compute_subgraph_iia(adj_matrix_kept, labels_kept, 0)
        print(f"  IIA for kept cluster {cluster_to_keep}: {cluster_iia:.4f}")
    else:
        print(f"\nSkipping graph building for kept cluster (only {len(filtered_by_query_group)} samples, need at least 2)")
        adj_matrix_kept = None
    
    # Build graph for filtered-out cluster
    if len(filtered_out_samples) >= 2:
        print(f"\nBuilding DIRECTED graph for filtered-out cluster (cluster {cluster_to_filter})...")
        adj_matrix_filtered = build_directed_graph(
            pipeline, causal_model, config, filtered_out_samples,
            args.layer, args.batch_size
        )
        
        # Compute IIA for filtered-out cluster
        labels_filtered = np.zeros(len(filtered_out_samples), dtype=int)
        filtered_out_iia = compute_subgraph_iia(adj_matrix_filtered, labels_filtered, 0)
        print(f"  IIA for filtered-out cluster {cluster_to_filter}: {filtered_out_iia:.4f}")
    else:
        print(f"\nSkipping graph building for filtered-out cluster (only {len(filtered_out_samples)} samples, need at least 2)")
        adj_matrix_filtered = None
    
    print(f"\nResults:")
    print(f"  Initial samples: {args.sample_size}")
    print(f"  After model filtering: {len(filtered_samples)}")
    print(f"  After query_group filtering (kept): {len(filtered_by_query_group)}")
    print(f"  After query_group filtering (filtered out): {len(filtered_out_samples)}")
    if cluster_iia is not None:
        print(f"  IIA for kept cluster {cluster_to_keep}: {cluster_iia:.4f}")
    if filtered_out_iia is not None:
        print(f"  IIA for filtered-out cluster {cluster_to_filter}: {filtered_out_iia:.4f}")
    
    # Save results
    results = {
        "task": args.task,
        "num_groups": args.num_groups,
        "model": args.model,
        "layer": args.layer,
        "partition_metadata": partition_metadata,
        "initial_sample_size": args.sample_size,
        "after_model_filtering": len(filtered_samples),
        "after_query_group_filtering_kept": len(filtered_by_query_group),
        "after_query_group_filtering_filtered_out": len(filtered_out_samples),
        "cluster_kept": int(cluster_to_keep),
        "cluster_filtered": int(cluster_to_filter),
        "iia_kept_cluster": float(cluster_iia) if cluster_iia is not None else None,
        "iia_filtered_out_cluster": float(filtered_out_iia) if filtered_out_iia is not None else None,
        "partition_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
    }
    
    # Create filename suffix from keep_indices
    keep_indices_str = "_".join(map(str, sorted(args.keep_indices)))
    
    results_path = output_dir / f"query_group_filter_results_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save filtered dataset (kept cluster)
    filtered_dataset_path = output_dir / f"filtered_by_query_group_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
    with open(filtered_dataset_path, 'wb') as f:
        pickle.dump(filtered_by_query_group, f)
    
    # Save filtered-out dataset
    filtered_out_dataset_path = output_dir / f"filtered_out_by_query_group_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
    with open(filtered_out_dataset_path, 'wb') as f:
        pickle.dump(filtered_out_samples, f)
    
    # Save graphs
    if adj_matrix_kept is not None:
        graph_path = output_dir / f"graph_query_group_filtered_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(adj_matrix_kept, f)
        print(f"Graph (kept cluster) saved to: {graph_path}")
    
    if adj_matrix_filtered is not None:
        graph_filtered_path = output_dir / f"graph_query_group_filtered_out_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}_keep_{keep_indices_str}.pkl"
        with open(graph_filtered_path, 'wb') as f:
            pickle.dump(adj_matrix_filtered, f)
        print(f"Graph (filtered-out cluster) saved to: {graph_filtered_path}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Filtered dataset (kept) saved to: {filtered_dataset_path}")
    print(f"Filtered dataset (filtered-out) saved to: {filtered_out_dataset_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
