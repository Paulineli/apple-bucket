#!/usr/bin/env python
"""
Graph Partitioning using Quasi-Clique Algorithm

This script:
1. Loads a graph from graph.pkl
2. Partitions the graph into K subgraphs by recursively finding K-1 quasi-cliques
   with threshold gamma (edge density >= gamma)
3. Leaves remaining nodes as the last cluster
4. Reports IIA (Interchange Intervention Accuracy) for each subgraph
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Set, Tuple
import numpy as np
from tqdm import tqdm


def compute_edge_density(adj_matrix: np.ndarray, nodes: List[int]) -> float:
    """Compute edge density of a subgraph induced by given nodes."""
    if len(nodes) <= 1:
        return 1.0  # Single node or empty set has density 1.0
    
    # Extract subgraph
    subgraph = adj_matrix[np.ix_(nodes, nodes)]
    
    # Count edges (divide by 2 for undirected graph)
    num_edges = np.sum(subgraph) // 2
    
    # Maximum possible edges
    n = len(nodes)
    max_edges = n * (n - 1) // 2
    
    return num_edges / max_edges if max_edges > 0 else 0.0


def find_quasi_clique_greedy(
    adj_matrix: np.ndarray,
    available_nodes: Set[int],
    gamma: float,
    min_size: int = 2
) -> Set[int]:
    """
    Find a quasi-clique with edge density >= gamma using a greedy algorithm.
    
    Algorithm:
    1. Start with a seed node (highest degree in available nodes)
    2. Iteratively add nodes that maintain density >= gamma
    3. Try multiple seeds and return the best quasi-clique
    """
    if len(available_nodes) < min_size:
        return set()
    
    available_list = list(available_nodes)
    best_quasi_clique = set()
    best_density = 0.0
    
    # Compute degrees within available nodes
    available_array = np.array(available_list)
    subgraph = adj_matrix[np.ix_(available_array, available_array)]
    degrees = np.sum(subgraph, axis=1)
    sorted_indices = np.argsort(degrees)[::-1]  # Sort by degree descending
    
    # Try top seeds (limit to avoid too many iterations)
    num_seeds = min(10, len(available_list))
    
    for seed_idx in sorted_indices[:num_seeds]:
        seed = available_list[seed_idx]
        quasi_clique = {seed}
        
        # Candidate nodes: all available nodes except current clique
        candidates = set(available_list) - quasi_clique
        
        # Greedily add nodes
        improved = True
        while improved and candidates:
            improved = False
            best_candidate = None
            best_new_density = 0.0
            
            # Try adding each candidate
            for candidate in list(candidates):
                test_clique = quasi_clique | {candidate}
                test_nodes = list(test_clique)
                density = compute_edge_density(adj_matrix, test_nodes)
                
                if density >= gamma and density > best_new_density:
                    best_new_density = density
                    best_candidate = candidate
                    improved = True
            
            if best_candidate is not None:
                quasi_clique.add(best_candidate)
                candidates.remove(best_candidate)
                # Optionally: filter candidates to only those connected to at least one node in clique
                # This speeds up the algorithm for large graphs
                if len(candidates) > 100:  # Only filter if many candidates
                    connected_to_clique = set()
                    for node in quasi_clique:
                        neighbors = set(np.where(adj_matrix[node])[0])
                        connected_to_clique |= neighbors
                    candidates = candidates & connected_to_clique
        
        # Check if this is the best quasi-clique found so far
        if len(quasi_clique) >= min_size:
            clique_nodes = list(quasi_clique)
            density = compute_edge_density(adj_matrix, clique_nodes)
            if density >= gamma and len(quasi_clique) > len(best_quasi_clique):
                best_quasi_clique = quasi_clique
                best_density = density
    
    return best_quasi_clique


def quasi_clique_partition(
    adj_matrix: np.ndarray,
    K: int,
    gamma: float,
    min_clique_size: int = 2
) -> np.ndarray:
    """
    Partition graph into K subgraphs by finding K-1 quasi-cliques.
    
    Algorithm:
    1. Recursively find K-1 quasi-cliques with density >= gamma
    2. Assign remaining nodes to the last cluster
    """
    n = adj_matrix.shape[0]
    labels = np.full(n, -1, dtype=int)  # -1 means unassigned
    
    available_nodes = set(range(n))
    cluster_id = 0
    
    print(f"Finding {K-1} quasi-cliques with gamma >= {gamma}...")
    
    # Find K-1 quasi-cliques
    for i in range(K - 1):
        if len(available_nodes) < min_clique_size:
            print(f"  Not enough nodes remaining for quasi-clique {i+1}. Stopping early.")
            break
        
        print(f"  Finding quasi-clique {i+1}/{K-1} (from {len(available_nodes)} available nodes)...")
        quasi_clique = find_quasi_clique_greedy(
            adj_matrix, available_nodes, gamma, min_clique_size
        )
        
        if len(quasi_clique) == 0:
            print(f"  Could not find quasi-clique {i+1} with gamma >= {gamma}. Stopping early.")
            break
        
        # Assign nodes in this quasi-clique to cluster
        for node in quasi_clique:
            labels[node] = cluster_id
        
        # Compute and report density
        clique_nodes = list(quasi_clique)
        density = compute_edge_density(adj_matrix, clique_nodes)
        print(f"    Found quasi-clique {i+1}: size={len(quasi_clique)}, density={density:.3f}")
        
        # Remove assigned nodes from available set
        available_nodes -= quasi_clique
        cluster_id += 1
    
    # Assign remaining nodes to the last cluster
    if available_nodes:
        print(f"  Assigning {len(available_nodes)} remaining nodes to cluster {cluster_id}...")
        for node in available_nodes:
            labels[node] = cluster_id
        cluster_id += 1
    
    # Handle any unassigned nodes (shouldn't happen, but just in case)
    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        print(f"  Warning: {len(unassigned)} nodes were unassigned. Assigning to cluster {cluster_id}...")
        for node in unassigned:
            labels[node] = cluster_id
    
    # Renumber clusters to be contiguous starting from 0
    unique_labels = np.unique(labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_mapping[label] for label in labels])
    
    return labels


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


def main():
    parser = argparse.ArgumentParser(description="Graph partitioning using quasi-clique algorithm")
    parser.add_argument("--graph-path", type=str, required=True, help="Path to graph.pkl file")
    parser.add_argument("--K", type=int, required=True, help="Number of subgraphs")
    parser.add_argument("--gamma", type=float, required=True, help="Minimum edge density threshold for quasi-cliques")
    parser.add_argument("--min-clique-size", type=int, default=2, help="Minimum size for a quasi-clique")
    parser.add_argument("--output-dir", type=str, default="./partition_results")
    parser.add_argument("--output-name", type=str, default=None, help="Output filename (without extension)")
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        adj_matrix = pickle.load(f)
    
    print(f"Graph loaded: {adj_matrix.shape[0]} nodes, {np.sum(adj_matrix) // 2} edges")
    
    # Compute overall IIA
    overall_iia = compute_overall_iia(adj_matrix)
    print(f"\nOverall IIA (edge density): {overall_iia:.3f}")
    
    # Partition using quasi-clique algorithm
    print(f"\nPartitioning graph into K={args.K} subgraphs...")
    labels = quasi_clique_partition(adj_matrix, args.K, args.gamma, args.min_clique_size)
    
    # Compute IIA for each subgraph
    print("\nSubgraph IIA (edge density):")
    unique_labels = np.unique(labels)
    iia_by_cluster = {}
    for k in unique_labels:
        iia = compute_subgraph_iia(adj_matrix, labels, k)
        iia_by_cluster[k] = iia
        cluster_size = np.sum(labels == k)
        print(f"  Cluster {k}: IIA = {iia:.3f} (size: {cluster_size})")
    
    # Save results
    results = {
        "method": "quasi_clique",
        "K": args.K,
        "gamma": args.gamma,
        "min_clique_size": args.min_clique_size,
        "overall_iia": overall_iia,
        "labels": labels.tolist(),
        "iia_by_cluster": {str(k): v for k, v in iia_by_cluster.items()},
    }
    
    # Determine output filename
    if args.output_name:
        output_filename = f"{args.output_name}.json"
    else:
        output_filename = f"partition_results_quasi_clique_K{args.K}_gamma{args.gamma:.2f}.json"
    
    results_path = output_dir / output_filename
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
