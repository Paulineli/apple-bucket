#!/usr/bin/env python
"""
Compute IIA (Interchange Intervention Accuracy) for clusters grouped by query_group index.

This script:
1. Loads the graph from graph.pkl
2. Loads samples to get query_group indices
3. Groups samples by query_group (6 groups: 0-5)
4. Computes IIA (edge density) for each query group cluster

Usage:
    python compute_iia_by_query_group.py [--graph-path PATH] [--samples-path PATH] [--num-groups 6] [--output PATH]

Example:
    python compute_iia_by_query_group.py --graph-path partition_results/graph.pkl --samples-path partition_results/filtered_input_samples.pkl
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _SCRIPT_DIR / "partition_results"


def compute_subgraph_iia(adj_matrix: np.ndarray, mask: np.ndarray) -> float:
    """Compute IIA (edge density) for a subgraph defined by a boolean mask."""
    subgraph = adj_matrix[mask][:, mask]
    n = subgraph.shape[0]
    if n <= 1:
        return 0.0
    num_edges = np.sum(subgraph) // 2  # Undirected graph
    max_edges = n * (n - 1) // 2
    return num_edges / max_edges if max_edges > 0 else 0.0


def compute_iia_by_query_group(
    graph_path: Path,
    samples_path: Path,
    num_groups: int = 6
) -> dict:
    """Compute IIA for each query group cluster."""
    
    # Load graph
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        adj_matrix = pickle.load(f)
    print(f"  Graph shape: {adj_matrix.shape}")
    
    # Load samples
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"  Number of samples: {len(samples)}")
    
    # Verify dimensions match
    if len(samples) != adj_matrix.shape[0]:
        raise ValueError(
            f"Dimension mismatch: {len(samples)} samples but graph has {adj_matrix.shape[0]} nodes"
        )
    
    # Extract query_group for each sample
    query_groups = [sample.get('query_group', None) for sample in samples]
    
    # Verify all samples have query_group
    if None in query_groups:
        raise ValueError("Some samples are missing 'query_group' field")
    
    # Group samples by query_group
    query_group_indices = defaultdict(list)
    for idx, qg in enumerate(query_groups):
        query_group_indices[qg].append(idx)
    
    print(f"\nQuery group distribution:")
    for qg in sorted(query_group_indices.keys()):
        print(f"  Query Group {qg}: {len(query_group_indices[qg])} samples")
    
    # Compute IIA for each query group
    results = {}
    print(f"\nComputing IIA for each query group cluster:")
    print("=" * 60)
    
    for qg in range(num_groups):
        if qg not in query_group_indices:
            print(f"Query Group {qg}: No samples (IIA = N/A)")
            results[qg] = {
                'iia': None,
                'num_samples': 0,
                'num_edges': 0,
                'max_edges': 0
            }
            continue
        
        indices = query_group_indices[qg]
        mask = np.zeros(len(samples), dtype=bool)
        mask[indices] = True
        
        iia = compute_subgraph_iia(adj_matrix, mask)
        
        # Compute additional statistics
        subgraph = adj_matrix[mask][:, mask]
        num_edges = np.sum(subgraph) // 2
        n = len(indices)
        max_edges = n * (n - 1) // 2
        
        results[qg] = {
            'iia': float(iia),
            'num_samples': n,
            'num_edges': int(num_edges),
            'max_edges': int(max_edges)
        }
        
        print(f"Query Group {qg}:")
        print(f"  IIA (edge density): {iia:.4f}")
        print(f"  Samples: {n}")
        print(f"  Edges: {num_edges} / {max_edges}")
        print()
    
    # Compute overall IIA for comparison
    overall_iia = compute_subgraph_iia(adj_matrix, np.ones(len(samples), dtype=bool))
    results['overall'] = {
        'iia': float(overall_iia),
        'num_samples': len(samples),
        'num_edges': int(np.sum(adj_matrix) // 2),
        'max_edges': int(len(samples) * (len(samples) - 1) // 2)
    }
    
    print("=" * 60)
    print(f"Overall IIA (entire graph): {overall_iia:.4f}")
    print(f"  Samples: {len(samples)}")
    print(f"  Edges: {results['overall']['num_edges']} / {results['overall']['max_edges']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute IIA for clusters grouped by query_group index"
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=_DEFAULT_RESULTS_DIR / "graph.pkl",
        help="Path to graph.pkl file",
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=_DEFAULT_RESULTS_DIR / "filtered_input_samples.pkl",
        help="Path to filtered_input_samples.pkl file",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=6,
        help="Number of query groups (default: 6)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: same dir as graph, iia_by_query_group.json)",
    )
    args = parser.parse_args()

    graph_path = args.graph_path
    samples_path = args.samples_path
    
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")
    
    # Compute IIA by query group
    results = compute_iia_by_query_group(graph_path, samples_path, args.num_groups)
    
    output_path = args.output if args.output is not None else graph_path.parent / "iia_by_query_group.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
