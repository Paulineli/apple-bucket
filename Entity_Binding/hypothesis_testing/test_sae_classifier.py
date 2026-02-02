#!/usr/bin/env python
"""
Filter Entity Binding Dataset Using Trained Classifier

This script:
1. Generates a dataset of entity binding samples (input only, no counterfactuals)
2. Uses a trained classifier to predict cluster membership (high vs low IIA)
3. Filters out samples predicted as low IIA cluster
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

from analyze_sae_features import (
    load_classifier,
    extract_activations_at_layer,
    load_sae,
    register_qwen3_for_pyvene,
)

# Register Qwen3 support
register_qwen3_for_pyvene()


def predict_cluster_with_classifier(
    pipeline,
    samples: List[Dict],
    classifier,
    classifier_metadata: Dict,
    sae,
    layer: int,
    batch_size: int,
    task_type: str,
    num_groups: int
) -> np.ndarray:
    """
    Use classifier to predict cluster membership for samples.
    
    Args:
        pipeline: LMPipeline object
        samples: List of input samples
        classifier: Trained LogisticRegression classifier
        classifier_metadata: Metadata dict with top_feature_indices (list or "all"),
            cluster_0_id, cluster_1_id, and n_features
        sae: SAE object
        layer: Layer number
        batch_size: Batch size for processing
        task_type: Task type
        num_groups: Number of groups
        
    Returns:
        Array of predicted cluster IDs (0 for cluster_0_id, 1 for cluster_1_id)
    """
    print("Extracting activations for classifier prediction...")
    
    # Extract activations
    activations = extract_activations_at_layer(
        pipeline,
        samples,
        layer,
        batch_size=batch_size,
        task_type=task_type,
        num_groups=num_groups
    )
    
    # Encode through SAE to get feature activations
    print("Encoding activations through SAE...")
    with torch.no_grad():
        device = next(sae.parameters()).device
        activations = activations.to(device)
        sae_output = sae.encode(activations)
        
        # Handle different return types
        if hasattr(sae_output, 'feature_acts'):
            feature_activations = sae_output.feature_acts
        elif isinstance(sae_output, torch.Tensor):
            feature_activations = sae_output
        elif isinstance(sae_output, tuple):
            feature_activations = sae_output[0]
        else:
            raise ValueError(f"Unexpected SAE output type: {type(sae_output)}")
    
    # Prepare features for classifier
    feature_activations_np = feature_activations.cpu().numpy()
    top_feature_indices = classifier_metadata["top_feature_indices"]
    
    if isinstance(top_feature_indices, list):
        # Top-k mode: use only selected SAE features
        X_input = feature_activations_np[:, top_feature_indices]
        print(f"Using top-k features for prediction (k={len(top_feature_indices)})")
    elif top_feature_indices == "all":
        # Full-L1 mode: use full SAE feature vector
        X_input = feature_activations_np
        print(f"Using full SAE feature vector for prediction (d={X_input.shape[1]})")
    else:
        raise ValueError(
            f"Unrecognized top_feature_indices in classifier metadata: {top_feature_indices}"
        )
    
    # Predict cluster membership
    print("Predicting cluster membership...")
    predictions_binary = classifier.predict(X_input)  # 0 or 1
    
    # Map back to original cluster IDs
    cluster_0_id = classifier_metadata["cluster_0_id"]
    cluster_1_id = classifier_metadata["cluster_1_id"]
    
    predictions = np.where(predictions_binary == 0, cluster_0_id, cluster_1_id)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Filter entity binding dataset using trained classifier"
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        required=True,
        help="Path to saved classifier (.pkl file)"
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
        default=6,
        help="Number of groups"
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="Path to SAE (HuggingFace model ID, local path, or 'release:sae_id' format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./classifier_filter_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--keep-low-iia",
        action="store_true",
        help="If set, keep low IIA cluster instead of filtering it out"
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
    print("Filter Entity Binding Dataset Using Classifier")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Classifier: {args.classifier_path}")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Device: {device}")
    print()
    
    # Load classifier
    print("Loading classifier...")
    classifier, classifier_metadata = load_classifier(args.classifier_path)
    print(f"  Loaded classifier with {classifier_metadata['n_features']} features")
    print(f"  Cluster 0 ID: {classifier_metadata['cluster_0_id']} (high IIA)")
    print(f"  Cluster 1 ID: {classifier_metadata['cluster_1_id']} (low IIA)")
    
    # Determine which cluster to keep
    cluster_to_keep = classifier_metadata['cluster_0_id']  # High IIA by default
    cluster_to_filter = classifier_metadata['cluster_1_id']  # Low IIA by default
    
    if args.keep_low_iia:
        cluster_to_keep, cluster_to_filter = cluster_to_filter, cluster_to_keep
        print(f"  Keeping cluster {cluster_to_keep} (low IIA)")
    else:
        print(f"  Keeping cluster {cluster_to_keep} (high IIA)")
    
    # Load model
    print(f"\nLoading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=5,
        device=device,
        max_length=200,
    )
    pipeline.tokenizer.padding_side = "left"
    print("  Model loaded")
    
    # Load SAE
    print(f"\nLoading SAE for layer {args.layer}...")
    sae = load_sae(args.model, args.layer, device=device, sae_path=args.sae_path)
    print("  SAE loaded")
    
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
    
    # Step 3: Predict cluster membership using classifier
    print(f"\nPredicting cluster membership for {len(filtered_samples)} samples...")
    predictions = predict_cluster_with_classifier(
        pipeline,
        filtered_samples,
        classifier,
        classifier_metadata,
        sae,
        args.layer,
        args.batch_size,
        args.task,
        args.num_groups
    )
    
    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cluster_id, count in zip(unique, counts):
        cluster_type = "high IIA" if cluster_id == classifier_metadata['cluster_0_id'] else "low IIA"
        print(f"  Cluster {cluster_id} ({cluster_type}): {count} samples")
    
    # Step 4: Filter to separate clusters
    keep_mask = predictions == cluster_to_keep
    filtered_by_classifier = [s for s, keep in zip(filtered_samples, keep_mask) if keep]
    filtered_out_samples = [s for s, keep in zip(filtered_samples, keep_mask) if not keep]
    
    print(f"\nFiltered to {len(filtered_by_classifier)} samples in cluster {cluster_to_keep}")
    print(f"Filtered out {len(filtered_out_samples)} samples in cluster {cluster_to_filter}")
    
    # Step 5: Build graphs and compute IIA for both clusters
    cluster_iia = None
    filtered_out_iia = None
    
    # Build graph for kept cluster
    if len(filtered_by_classifier) >= 2:
        print(f"\nBuilding DIRECTED graph for kept cluster (cluster {cluster_to_keep})...")
        adj_matrix_kept = build_directed_graph(
            pipeline, causal_model, config, filtered_by_classifier,
            args.layer, args.batch_size
        )
        
        # Compute IIA for kept cluster
        labels_kept = np.zeros(len(filtered_by_classifier), dtype=int)
        cluster_iia, cluster_undirected_density = compute_directed_and_undirected_density_from_directed(adj_matrix_kept)
        print(f"  IIA for kept cluster {cluster_to_keep}: {cluster_iia:.4f}")
        print(f"  Undirected density for kept cluster {cluster_to_keep}: {cluster_undirected_density:.4f}")
    else:
        print(f"\nSkipping graph building for kept cluster (only {len(filtered_by_classifier)} samples, need at least 2)")
        adj_matrix_kept = None
    
    # Build graph for filtered-out cluster
    if len(filtered_out_samples) >= 2:
        print(f"\nBuilding DIRECTED graph for filtered-out cluster (cluster {cluster_to_filter})...")
        adj_matrix_filtered = build_directed_graph(
            pipeline, causal_model, config, filtered_out_samples,
            args.layer, args.batch_size
        )
        
        # Compute IIA for filtered-out cluster
        # labels_filtered = np.zeros(len(filtered_out_samples), dtype=int)
        filtered_out_iia, filtered_out_undirected_density = compute_directed_and_undirected_density_from_directed(adj_matrix_filtered)
        print(f"  IIA for filtered-out cluster {cluster_to_filter}: {filtered_out_iia:.4f}")
        print(f"  Undirected density for filtered-out cluster {cluster_to_filter}: {filtered_out_undirected_density:.4f}")
    else:
        print(f"\nSkipping graph building for filtered-out cluster (only {len(filtered_out_samples)} samples, need at least 2)")
        adj_matrix_filtered = None
    
    print(f"\nResults:")
    print(f"  Initial samples: {args.sample_size}")
    print(f"  After model filtering: {len(filtered_samples)}")
    print(f"  After classifier filtering (kept): {len(filtered_by_classifier)}")
    print(f"  After classifier filtering (filtered out): {len(filtered_out_samples)}")
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
        "classifier_path": args.classifier_path,
        "classifier_metadata": classifier_metadata,
        "initial_sample_size": args.sample_size,
        "after_model_filtering": len(filtered_samples),
        "after_classifier_filtering_kept": len(filtered_by_classifier),
        "after_classifier_filtering_filtered_out": len(filtered_out_samples),
        "cluster_kept": int(cluster_to_keep),
        "cluster_filtered": int(cluster_to_filter),
        "iia_kept_cluster": float(cluster_iia) if cluster_iia is not None else None,
        "iia_filtered_out_cluster": float(filtered_out_iia) if filtered_out_iia is not None else None,
        "undirected_density_kept_cluster": float(cluster_undirected_density) if cluster_undirected_density is not None else None,
        "undirected_density_filtered_out_cluster": float(filtered_out_undirected_density) if filtered_out_undirected_density is not None else None,
        "prediction_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
    }
    
    results_path = output_dir / f"classifier_filter_results_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save filtered dataset (kept cluster)
    filtered_dataset_path = output_dir / f"filtered_by_classifier_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
    with open(filtered_dataset_path, 'wb') as f:
        pickle.dump(filtered_by_classifier, f)
    
    # Save filtered-out dataset
    filtered_out_dataset_path = output_dir / f"filtered_out_by_classifier_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
    with open(filtered_out_dataset_path, 'wb') as f:
        pickle.dump(filtered_out_samples, f)
    
    # Save graphs
    if adj_matrix_kept is not None:
        graph_path = output_dir / f"graph_classifier_filtered_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(adj_matrix_kept, f)
        print(f"Graph (kept cluster) saved to: {graph_path}")
    
    if adj_matrix_filtered is not None:
        graph_filtered_path = output_dir / f"graph_classifier_filtered_out_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
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
