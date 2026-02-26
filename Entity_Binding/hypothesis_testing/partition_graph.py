#!/usr/bin/env python
"""
Graph Partitioning for Entity Binding Samples

This script:
1. Generates a dataset of entity binding samples (input only, no counterfactuals)
2. Builds a graph where edges represent consistent interchange interventions
3. Partitions the graph using the quasi-clique algorithm (see partition_graph_quasi_clique.py)
4. Reports IIA (Interchange Intervention Accuracy) for each subgraph
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
sys.path.insert(0, os.path.join(_script_dir, '..', '..', 'causalab'))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from partition_graph_quasi_clique import quasi_clique_partition

from causalab.tasks.entity_binding.config import (
    EntityBindingTaskConfig,
    create_sample_action_config,
    create_filling_liquids_config,
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import build_token_position_factories
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.metric import causal_score_intervention_outputs
from causalab.experiments.filter import filter_dataset
from causalab.causal.counterfactual_dataset import CounterfactualExample


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


def create_config(task_type: str, num_groups: int) -> EntityBindingTaskConfig:
    """Create task config."""
    if task_type == "filling_liquids":
        config = create_filling_liquids_config()
    else:
        config = create_sample_action_config()
    
    config.max_groups = num_groups
    config.prompt_prefix = "We will ask a question about the following sentences. Only return the answer, no other text.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"
    
    return config


def generate_input_samples(
    config: EntityBindingTaskConfig,
    causal_model,
    size: int,
    query_index: int,
    answer_index: int,
    num_groups: int
) -> List[Dict]:
    """Generate input-only samples with fixed query/answer indices but random query_group."""
    config.fixed_query_indices = (query_index,)
    config.fixed_answer_index = answer_index
    config.fixed_active_groups = num_groups
    
    samples = []
    for _ in range(size):
        trace = sample_valid_entity_binding_input(config, causal_model, ensure_positional_uniqueness=True)
        # Convert CausalTrace to dict (input variables only)
        sample = {var: trace[var] for var in causal_model.inputs}
        samples.append(sample)
    
    return samples


def checker(neural_output, causal_output):
    """Check if neural output matches causal output."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


def filter_input_samples(
    samples: List[Dict],
    pipeline: LMPipeline,
    causal_model,
    batch_size: int
) -> List[Dict]:
    """Filter input samples to keep only those where model performs perfectly."""
    print(f"Filtering {len(samples)} input samples...")
    
    # Convert samples to CounterfactualExamples (input only, no counterfactuals)
    cf_examples: List[CounterfactualExample] = []
    for sample in samples:
        trace = causal_model.new_trace(sample)
        cf_examples.append({
            "input": trace,
            "counterfactual_inputs": []
        })
    
    # Filter dataset (only validates base inputs since no counterfactuals)
    filtered_cf_examples = filter_dataset(
        dataset=cf_examples,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=batch_size,
        validate_counterfactuals=False
    )
    
    # Convert back to dicts
    filtered_samples = []
    for cf_example in filtered_cf_examples:
        trace = cf_example["input"]
        sample = {var: trace[var] for var in causal_model.inputs}
        filtered_samples.append(sample)
    
    print(f"  Kept {len(filtered_samples)}/{len(samples)} samples ({len(filtered_samples)/len(samples)*100:.1f}%)")
    
    return filtered_samples


def create_pair_counterfactuals(
    traces: List,
    pairs: List[Tuple[int, int]]
) -> Tuple[List[CounterfactualExample], List[Tuple[int, int, str]]]:
    """Create counterfactual examples for all pairs in both directions."""
    all_cf_examples: List[CounterfactualExample] = []
    pair_indices = []
    
    for i, j in pairs:
        trace_i = traces[i]
        trace_j = traces[j]
        
        # (i->j) direction
        all_cf_examples.append({
            "input": trace_i,
            "counterfactual_inputs": [trace_j]
        })
        pair_indices.append((i, j, "ij"))
        
        # (j->i) direction
        all_cf_examples.append({
            "input": trace_j,
            "counterfactual_inputs": [trace_i]
        })
        pair_indices.append((i, j, "ji"))
    
    return all_cf_examples, pair_indices


def compute_per_example_scores(
    raw_results: Dict,
    all_cf_examples: List[CounterfactualExample],
    causal_model
) -> List[float]:
    """Compute per-example consistency scores from raw intervention results."""
    # Label dataset with causal model expectations
    labeled_data = causal_model.label_counterfactual_data(
        all_cf_examples,
        ["positional_answer"]
    )
    
    # Extract and flatten string outputs
    string_outputs = raw_results["target"].get("string", [])
    flattened_outputs = []
    for item in string_outputs:
        if isinstance(item, list):
            flattened_outputs.extend(item)
        else:
            flattened_outputs.append(item)
    
    # Compute scores
    scores = []
    for idx, output_string in enumerate(flattened_outputs):
        if idx < len(labeled_data):
            expected = labeled_data[idx]["label"]
            expected_str = expected.get("string", str(expected)) if isinstance(expected, dict) else str(expected)
            neural_output = {"string": output_string}
            is_consistent = checker(neural_output, expected_str)
            scores.append(1.0 if is_consistent else 0.0)
        else:
            scores.append(0.0)
    
    return scores


def build_adjacency_from_scores(
    scores: List[float],
    pair_indices: List[Tuple[int, int, str]],
    n: int
) -> np.ndarray:
    """Build UNDIRECTED adjacency matrix from per-example scores.

    An undirected edge between i and j exists only if BOTH directed
    interventions (i -> j) and (j -> i) are correct.
    """
    adj_matrix = np.zeros((n, n), dtype=bool)
    
    print("Processing results and building adjacency matrix...")
    for idx, (i, j, direction) in enumerate(tqdm(pair_indices, desc="Building graph")):
        if direction == "ij":
            score_ij = scores[idx]
            score_ji = scores[idx + 1] if idx + 1 < len(scores) else 0.0
            
            # Edge exists if both directions are consistent
            if score_ij > 0.5 and score_ji > 0.5:
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True
    
    return adj_matrix


def build_directed_adjacency_from_scores(
    scores: List[float],
    pair_indices: List[Tuple[int, int, str]],
    n: int
) -> np.ndarray:
    """Build DIRECTED adjacency matrix from per-example scores.

    Here we keep full directional information:
      - If (i -> j) is correct, we set adj[i, j] = True
      - If (j -> i) is correct, we set adj[j, i] = True

    Thus a bidirected edge (both directions correct) will have both
    adj[i, j] and adj[j, i] set to True, whereas a unidirected edge
    will have only one of them set.
    """
    adj_matrix = np.zeros((n, n), dtype=bool)

    print("Processing results and building DIRECTED adjacency matrix...")
    for idx, (i, j, direction) in enumerate(tqdm(pair_indices, desc="Building directed graph")):
        score = scores[idx]
        if score > 0.5:
            if direction == "ij":
                adj_matrix[i, j] = True
            elif direction == "ji":
                adj_matrix[j, i] = True

    return adj_matrix


def build_graph(
    pipeline: LMPipeline,
    causal_model,
    config: EntityBindingTaskConfig,
    samples: List[Dict],
    layer: int,
    batch_size: int
) -> np.ndarray:
    """Build graph adjacency matrix based on interchange intervention consistency."""
    n = len(samples)
    
    # Setup token positions and target
    template = config.build_mega_template(
        active_groups=config.max_groups,
        query_indices=config.fixed_query_indices,
        answer_index=config.fixed_answer_index,
    )
    token_position_specs = {"last_token": {"type": "index", "position": -1}}
    factories = build_token_position_factories(token_position_specs, template)
    token_positions = [list(factories.values())[0](pipeline)]
    
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=[layer],
        token_positions=token_positions,
        mode="one_target_per_unit",
    )
    target = list(targets.values())[0]
    
    # Generate pairs and create counterfactual examples
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"Building graph: checking {len(pairs)} pairs (2 interventions per pair)...")
    
    traces = [causal_model.new_trace(sample) for sample in samples]
    all_cf_examples, pair_indices = create_pair_counterfactuals(traces, pairs)
    
    # Run interventions
    num_batches = (len(all_cf_examples) + batch_size - 1) // batch_size
    print(f"Running {len(all_cf_examples)} interchange interventions in {num_batches} batches...")
    
    logger = logging.getLogger("causalab.neural.pyvene_core.interchange")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    try:
        raw_results = {
            "target": run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=all_cf_examples,
                interchange_target=target,
                batch_size=batch_size,
                output_scores=False,
            )
        }
    finally:
        logger.setLevel(old_level)
    
    # Compute scores and build adjacency matrix
    scores = compute_per_example_scores(raw_results, all_cf_examples, causal_model)
    adj_matrix = build_adjacency_from_scores(scores, pair_indices, n)
    
    return adj_matrix


def build_directed_graph(
    pipeline: LMPipeline,
    causal_model,
    config: EntityBindingTaskConfig,
    samples: List[Dict],
    layer: int,
    batch_size: int
) -> np.ndarray:
    """Build DIRECTED graph adjacency matrix based on interchange intervention consistency.

    This is identical to `build_graph` except that it preserves directionality:
    an edge i -> j exists if the (i -> j) intervention has the correct outcome,
    independently of whether (j -> i) is also correct.
    """
    n = len(samples)
    
    # Setup token positions and target
    template = config.build_mega_template(
        active_groups=config.max_groups,
        query_indices=config.fixed_query_indices,
        answer_index=config.fixed_answer_index,
    )
    token_position_specs = {"last_token": {"type": "index", "position": -1}}
    factories = build_token_position_factories(token_position_specs, template)
    token_positions = [list(factories.values())[0](pipeline)]
    
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=[layer],
        token_positions=token_positions,
        mode="one_target_per_unit",
    )
    target = list(targets.values())[0]
    
    # Generate pairs and create counterfactual examples
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"Building DIRECTED graph: checking {len(pairs)} pairs (2 interventions per pair)...")
    
    traces = [causal_model.new_trace(sample) for sample in samples]
    all_cf_examples, pair_indices = create_pair_counterfactuals(traces, pairs)
    
    # Run interventions
    num_batches = (len(all_cf_examples) + batch_size - 1) // batch_size
    print(f"Running {len(all_cf_examples)} interchange interventions in {num_batches} batches...")
    
    logger = logging.getLogger("causalab.neural.pyvene_core.interchange")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    try:
        raw_results = {
            "target": run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=all_cf_examples,
                interchange_target=target,
                batch_size=batch_size,
                output_scores=False,
            )
        }
    finally:
        logger.setLevel(old_level)
    
    # Compute scores and build DIRECTED adjacency matrix
    scores = compute_per_example_scores(raw_results, all_cf_examples, causal_model)
    adj_matrix = build_directed_adjacency_from_scores(scores, pair_indices, n)
    
    return adj_matrix


def compute_overall_iia(adj_matrix: np.ndarray) -> float:
    """Compute overall IIA (edge density) for the entire graph."""
    n = adj_matrix.shape[0]
    if n <= 1:
        return 0.0
    num_edges = np.sum(adj_matrix) // 2  # Undirected graph
    max_edges = n * (n - 1) // 2
    return num_edges / max_edges if max_edges > 0 else 0.0


def compute_subgraph_iia(adj_matrix: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    """Compute IIA (edge density) for a subgraph."""
    mask = labels == cluster_id
    subgraph = adj_matrix[mask][:, mask]
    n = subgraph.shape[0]
    if n <= 1:
        return 0.0
    num_edges = np.sum(subgraph) // 2  # Undirected graph
    max_edges = n * (n - 1) // 2
    return num_edges / max_edges if max_edges > 0 else 0.0


def compute_directed_and_undirected_density_from_directed(
    directed_adj_matrix: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute edge densities for a DIRECTED graph and the corresponding UNDIRECTED
    graph obtained by keeping only bidirected edges.

    - Directed density counts every existing directed edge (i -> j, i != j).
    - Undirected density first constructs an undirected adjacency where an edge
      between i and j exists only if BOTH i -> j and j -> i are present, and
      then computes the usual undirected edge density.
    """
    n = directed_adj_matrix.shape[0]
    if n <= 1:
        return 0.0, 0.0

    # Remove any self-loops from consideration, just in case.
    directed_no_self = directed_adj_matrix.copy()
    np.fill_diagonal(directed_no_self, False)

    # Directed density: count all directed edges.
    num_directed_edges = float(np.sum(directed_no_self))
    max_directed_edges = float(n * (n - 1))
    directed_density = num_directed_edges / max_directed_edges if max_directed_edges > 0 else 0.0

    # Undirected density: keep only bidirected edges.
    bidirected = np.logical_and(directed_no_self, directed_no_self.T)
    np.fill_diagonal(bidirected, False)
    num_undirected_edges = float(np.sum(bidirected)) / 2.0
    max_undirected_edges = float(n * (n - 1) // 2)
    undirected_density = (
        num_undirected_edges / max_undirected_edges if max_undirected_edges > 0 else 0.0
    )

    return directed_density, undirected_density


def main():
    parser = argparse.ArgumentParser(description="Graph partitioning for entity binding")
    parser.add_argument("--task", type=str, default="action", choices=["action", "filling_liquids"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--K", type=int, required=True, help="Number of subgraphs")
    parser.add_argument("--query-index", type=int, default=1)
    parser.add_argument("--answer-index", type=int, default=0)
    parser.add_argument("--num-groups", type=int, default=6)
    parser.add_argument("--graph-path", type=str, default=None, help="Path to saved graph")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to saved dataset")
    parser.add_argument("--output-dir", type=str, default="./partition_results")
    parser.add_argument("--gamma", type=float, default=0.9, help="Minimum edge density for quasi-cliques")
    parser.add_argument("--min-clique-size", type=int, default=2, help="Minimum size for a quasi-clique")
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # After setting CUDA_VISIBLE_DEVICES, the visible device is re-indexed to 0
        device = "cuda:0"
    else:
        device = "cpu"
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "_")  # e.g. google/gemma-2-2b-it -> google-gemma-2-2b-it
    
    # Load model
    print(f"Loading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=5,
        device=device,
        max_length=200,
    )
    pipeline.tokenizer.padding_side = "left"
    
    # Create config and causal model
    config = create_config(args.task, args.num_groups)
    causal_model = create_positional_entity_causal_model(config)
    
    if args.dataset_path and not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    if args.graph_path and not Path(args.graph_path).exists():
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")

    # Step 1: Generate or load dataset
    if args.dataset_path and Path(args.dataset_path).exists():
        print(f"Loading dataset from {args.dataset_path}...")
        with open(args.dataset_path, 'rb') as f:
            samples = pickle.load(f)
    else:
        print(f"Generating {args.sample_size} input samples...")
        samples = generate_input_samples(
            config, causal_model, args.sample_size,
            args.query_index, args.answer_index, args.num_groups
        )
    
        samples = filter_input_samples(samples, pipeline, causal_model, args.batch_size)
    
        if len(samples) == 0:
            raise ValueError("No samples passed filtering! Model cannot perform correctly on any samples.")
        
        # Save filtered dataset
        filtered_dataset_path = output_dir / f"filtered_input_samples_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
        with open(filtered_dataset_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved filtered dataset to {filtered_dataset_path}")
    
    # Step 2: Build or load graph
    if args.graph_path and Path(args.graph_path).exists():
        print(f"Loading graph from {args.graph_path}...")
        with open(args.graph_path, 'rb') as f:
            adj_matrix = pickle.load(f)
    else:
        print("Building graph from interchange interventions...")
        adj_matrix = build_graph(
            pipeline, causal_model, config, samples,
            args.layer, args.batch_size
        )
        graph_path = output_dir / f"graph_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(adj_matrix, f)
        print(f"Saved graph to {graph_path}")
    
    # Step 3: Quasi-clique partition
    print(f"Partitioning graph with quasi-clique algorithm (K={args.K}, gamma={args.gamma})...")
    labels = quasi_clique_partition(
        adj_matrix, args.K, args.gamma, args.min_clique_size
    )
    
    # Compute overall IIA
    overall_iia = compute_overall_iia(adj_matrix)
    print(f"\nOverall IIA (edge density): {overall_iia:.3f}")
    
    # Compute IIA for each subgraph
    print("\nSubgraph IIA (edge density):")
    unique_labels = np.unique(labels)
    iia_by_cluster = {}
    for k in unique_labels:
        iia = compute_subgraph_iia(adj_matrix, labels, int(k))
        iia_by_cluster[int(k)] = iia
        cluster_size = np.sum(labels == k)
        print(f"  Cluster {k}: IIA = {iia:.3f} (size: {cluster_size})")
    
    # Save results
    results = {
        "task": args.task,
        "num_groups": args.num_groups,
        "model": args.model,
        "layer": args.layer,
        "sample_size": len(samples),
        "method": "quasi_clique",
        "K": args.K,
        "gamma": args.gamma,
        "min_clique_size": args.min_clique_size,
        "overall_iia": overall_iia,
        "labels": labels.tolist(),
        "iia_by_cluster": iia_by_cluster,
    }
    
    results_path = output_dir / f"partition_results_{args.task}_{args.layer}_{args.num_groups}_{args.sample_size}_{model_safe}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
