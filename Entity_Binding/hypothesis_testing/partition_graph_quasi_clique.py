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
import time
from pathlib import Path
from typing import List, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm
import random


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


def _edges_in_subgraph(adj_matrix: np.ndarray, nodes: List[int]) -> int:
    """Number of edges in the subgraph induced by nodes (undirected)."""
    if len(nodes) <= 1:
        return 0
    sub = adj_matrix[np.ix_(nodes, nodes)]
    return int(np.sum(sub) // 2)


def _count_edges_to_set(adj_matrix: np.ndarray, v: int, S: List[int]) -> int:
    """Number of edges from node v to set S."""
    return int(np.sum(adj_matrix[v, S]))


def find_quasi_clique_branch_and_bound(
    adj_matrix: np.ndarray,
    available_nodes: Set[int],
    gamma: float,
    min_size: int = 2,
    time_limit: Optional[float] = None,
    node_limit: Optional[int] = None,
) -> Set[int]:
    """
    Find a maximum-size quasi-clique with edge density >= gamma using Branch and Bound.
    Single entry point recurse([], all_nodes); at each node pick one pivot and branch
    exactly twice (Include / Exclude). Uses only valid pruning: UB = |S| + |R| (no
    quasi-clique in S∪R can be larger).

    When time_limit (seconds) or node_limit (max BnB nodes explored) is set, search
    stops at the limit and returns the best solution found so far (exact otherwise).
    """
    if len(available_nodes) < min_size:
        return set()
    available_list = list(available_nodes)

    best_solution: List[int] = []
    best_size = [0]  # use list to allow mutation in recursion
    start_time = time.perf_counter()
    nodes_explored = [0]  # mutable counter
    stop = [False]  # set when limit hit

    def recurse(S: List[int], R: List[int]) -> None:
        if stop[0]:
            return
        if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
            stop[0] = True
            return
        if node_limit is not None:
            nodes_explored[0] += 1
            if nodes_explored[0] > node_limit:
                stop[0] = True
                return
        # Evaluate current set: quasi-cliques are not hereditary
        if len(S) >= min_size and compute_edge_density(adj_matrix, S) >= gamma:
            if len(S) > best_size[0]:
                best_size[0] = len(S)
                best_solution[:] = S
        if not R:
            return
        # Valid prune: any quasi-clique in S∪R has size at most |S| + |R|
        if len(S) + len(R) <= best_size[0]:
            return
        # Standard BnB: one pivot, branch exactly twice (Include / Exclude)
        R_sorted = sorted(
            R,
            key=lambda v: _count_edges_to_set(adj_matrix, v, S),
            reverse=True,
        )
        pivot = R_sorted[0]
        R_rest = R_sorted[1:]
        # Include first: find large quasi-cliques early to raise best_size and improve pruning
        recurse(S + [pivot], R_rest)
        recurse(S, R_rest)

    recurse([], available_list)
    return set(best_solution) if best_solution else set()


def find_quasi_clique_gurobi(
    adj_matrix: np.ndarray,
    available_nodes: Set[int],
    gamma: float,
    min_size: int = 2,
    time_limit: Optional[float] = None,
) -> Set[int]:
    """
    Find a maximum-size quasi-clique with edge density >= gamma using Gurobi (exact MIQP).
    Formulation: max sum_i x_i  s.t.  sum_{i<j} A_ij x_i x_j >= gamma * (sum x_i)*(sum x_i - 1)/2.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi (gurobipy) is required for find_quasi_clique_gurobi. Install with: pip install gurobipy")

    if len(available_nodes) < min_size:
        return set()
    nodes = sorted(available_nodes)
    n = len(nodes)
    # Extract subgraph adjacency (indices 0..n-1)
    A = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if u != v and adj_matrix[u, v]:
                A[i, j] = 1

    m = gp.Model("quasi_clique")
    m.setParam("OutputFlag", 0)
    # Non-convex: constraint 2*Q >= gamma*s*(s-1) is indefinite quadratic; use spatial BnB
    m.setParam("NonConvex", 2)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)

    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    # Objective: maximize size
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    # Linearize: y[i,j] = x[i]*x[j] for i < j. Then edges = sum_{i<j} A[i,j]*y[i,j]
    # and size s = sum_i x[i], we need 2*edges >= gamma * s*(s-1).
    # s*(s-1) = sum_i x[i]*(x[i]-1) + 2*sum_{i<j} x[i]x[j] = 2*sum_{i<j} x[i]x[j] (since x[i]^2 = x[i])
    # So s^2 - s = 2*sum_{i<j} x[i]x[j], so sum_{i<j} x[i]x[j] = (s^2 - s)/2.
    # Constraint: 2*sum_{i<j} A[i,j] x[i]x[j] >= gamma * (s^2 - s)  =>  sum A[i,j]x[i]x[j] >= gamma*(s^2-s)/2.
    # We use auxiliary variables: s = sum_i x[i], and Q = sum_{i<j} A[i,j] x[i]x[j].
    # Constraint: 2*Q >= gamma * s*(s-1). Gurobi can handle bilinear terms.
    s = m.addVar(lb=0, ub=n, vtype=GRB.CONTINUOUS, name="s")
    m.addConstr(s == gp.quicksum(x[i] for i in range(n)))
    # Q = sum_{i<j} A[i,j] x[i] x[j]. We need 2*Q >= gamma * s*(s-1).
    # Gurobi 10+ supports general quadratic constraints. Add 2*Q - gamma*s*(s-1) >= 0.
    Q_expr = gp.QuadExpr()
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                Q_expr += x[i] * x[j]
    m.addConstr(2 * Q_expr >= gamma * s * (s - 1), "density")
    # Minimum size
    m.addConstr(s >= min_size)

    m.optimize()
    # Only read .X when a solution exists (optimal or incumbent at time limit)
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return set()
    if m.SolCount == 0:
        return set()
    try:
        sol = [nodes[i] for i in range(n) if x[i].X >= 0.5]
    except (AttributeError, gp.GurobiError):
        return set()
    return set(sol)


def find_quasi_clique_rls(
    adj_matrix: np.ndarray,
    available_nodes: Set[int],
    gamma: float,
    min_size: int = 2,
    max_steps: int = 5000,
    steps_no_improve_restart: int = 500,
    recent_visit_threshold: int = 50,
    steps_to_decrease_t: int = 100,
    T_init: int = 1,
    T_min: int = 1,
    T_max: int = 50,
    seed: Optional[int] = None,
) -> Set[int]:
    """
    Reactive Local Search (RLS) for maximum quasi-clique with edge density >= gamma.
    Uses add/remove moves, Tabu (prohibition T), reactive T adjustment, and restarts.
    """
    if seed is not None:
        random.seed(seed)
    if len(available_nodes) < min_size:
        return set()
    available_list = list(available_nodes)

    def edges_to_set(v: int, V: Set[int]) -> int:
        """Number of edges from node v to set V (efficient: O(|V|))."""
        return _count_edges_to_set(adj_matrix, v, list(V))

    # Track visited states (frozenset) for reactive mechanism
    # Note: For large graphs, exact state matches are rare; reactive T adjustment
    # may trigger less often. Alternative: track objective value cycles.
    visited_at: dict = {}  # frozenset(clique) -> last step when it was current
    best_clique: Set[int] = set()
    best_size = 0
    used_as_restart: Set[int] = set()

    def is_quasi_clique(V: Set[int]) -> bool:
        """Check if V is a valid quasi-clique (only used when updating best_clique)."""
        if len(V) < min_size:
            return False
        return compute_edge_density(adj_matrix, list(V)) >= gamma

    step = 0
    # Start from a random single-node quasi-clique (any single node has density 1)
    current = set([random.choice(available_list)])
    T = T_init
    tabu_add: dict = {}  # node -> step until which it's tabu for add
    tabu_remove: dict = {}  # node -> step until which it's tabu for remove
    last_improvement_step = 0
    last_T_increase_step = 0

    while step < max_steps:
        step += 1
        current_frozen = frozenset(current)
        # Memory reaction: if this clique was visited too recently, increase T (check before updating)
        if current_frozen in visited_at:
            prev_visit = visited_at[current_frozen]
            if step - prev_visit < recent_visit_threshold:
                T = min(T_max, T + 1)
                last_T_increase_step = step
        visited_at[current_frozen] = step

        # Update best_clique only when current is a valid quasi-clique
        if len(current) >= min_size and is_quasi_clique(current):
            if len(current) > best_size:
                best_size = len(current)
                best_clique = set(current)
                last_improvement_step = step

        # Pulse strategy: Dense (valid quasi-clique) -> GROW (add); Sparse (invalid) -> SHRINK (remove).
        # When |current| < min_size we are building up -> prefer ADD so we can reach a valid clique.
        dense = len(current) >= min_size and is_quasi_clique(current)
        building = len(current) < min_size
        candidates_add = [
            v for v in available_list
            if v not in current and step > tabu_add.get(v, 0)
        ]
        candidates_remove = [
            v for v in current
            if step > tabu_remove.get(v, 0)
        ]

        if (dense or building) and candidates_add:
            # Valid quasi-clique: try to grow by adding best node
            candidates_with_score = [
                (v, edges_to_set(v, current)) for v in candidates_add
            ]
            best_val = max(c[1] for c in candidates_with_score)
            best_cands = [c[0] for c in candidates_with_score if c[1] == best_val]
            chosen = random.choice(best_cands)
            current.add(chosen)
            tabu_remove[chosen] = step + T
        elif not dense and not building and candidates_remove:
            # Invalid (sparse): shrink by removing worst node
            candidates_with_score = [
                (v, edges_to_set(v, current - {v})) for v in candidates_remove
            ]
            best_val = min(c[1] for c in candidates_with_score)
            best_cands = [c[0] for c in candidates_with_score if c[1] == best_val]
            chosen = random.choice(best_cands)
            current.discard(chosen)
            tabu_add[chosen] = step + T
        elif not dense and not building and not candidates_remove and current:
            # Sparse but all tabu for remove: remove at random
            chosen = random.choice(list(current))
            current.discard(chosen)
            tabu_add[chosen] = step + T
        elif dense and not candidates_add and candidates_remove:
            # Dense but all tabu for add: remove to allow diversification
            candidates_with_score = [
                (v, edges_to_set(v, current - {v})) for v in candidates_remove
            ]
            best_val = min(c[1] for c in candidates_with_score)
            best_cands = [c[0] for c in candidates_with_score if c[1] == best_val]
            chosen = random.choice(best_cands)
            current.discard(chosen)
            tabu_add[chosen] = step + T
        elif current:
            # Fallback: remove at random
            chosen = random.choice(list(current))
            current.discard(chosen)
            tabu_add[chosen] = step + T

        # Decrease T if no recent increase (diversification was sufficient)
        if step - last_T_increase_step >= steps_to_decrease_t:
            T = max(T_min, T - 1)
            last_T_increase_step = step

        # Restart
        if step - last_improvement_step >= steps_no_improve_restart:
            unused = [u for u in available_list if u not in used_as_restart]
            if unused:
                start_node = random.choice(unused)
                used_as_restart.add(start_node)
            else:
                start_node = random.choice(available_list)
            current = set([start_node])
            T = T_init
            visited_at.clear()
            tabu_add.clear()
            tabu_remove.clear()
            last_improvement_step = step

    return best_clique if best_clique else set()


QUASI_CLIQUE_METHODS = {
    "greedy": find_quasi_clique_greedy,
    "branch_and_bound": find_quasi_clique_branch_and_bound,
    "gurobi": find_quasi_clique_gurobi,
    "rls": find_quasi_clique_rls,
}


def quasi_clique_partition(
    adj_matrix: np.ndarray,
    K: int,
    gamma: float,
    min_clique_size: int = 2,
    method: str = "greedy",
    **method_kwargs,
) -> np.ndarray:
    """
    Partition graph into K subgraphs by finding K-1 quasi-cliques.
    
    Algorithm:
    1. Recursively find K-1 quasi-cliques with density >= gamma
    2. Assign remaining nodes to the last cluster

    method: one of 'greedy', 'branch_and_bound', 'gurobi', 'rls'.
    method_kwargs: passed to the quasi-clique finder (e.g. time_limit for gurobi, max_steps for rls).
    """
    n = adj_matrix.shape[0]
    labels = np.full(n, -1, dtype=int)  # -1 means unassigned
    
    available_nodes = set(range(n))
    cluster_id = 0
    finder = QUASI_CLIQUE_METHODS.get(method)
    if finder is None:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(QUASI_CLIQUE_METHODS.keys())}")
    
    print(f"Finding {K-1} quasi-cliques with gamma >= {gamma} (method={method})...")
    
    # Find K-1 quasi-cliques
    for i in range(K - 1):
        if len(available_nodes) < min_clique_size:
            print(f"  Not enough nodes remaining for quasi-clique {i+1}. Stopping early.")
            break
        
        print(f"  Finding quasi-clique {i+1}/{K-1} (from {len(available_nodes)} available nodes)...")
        quasi_clique = finder(
            adj_matrix, available_nodes, gamma, min_clique_size, **method_kwargs
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
    parser.add_argument("--method", type=str, default="greedy", choices=list(QUASI_CLIQUE_METHODS.keys()),
                        help="Quasi-clique algorithm: greedy, branch_and_bound, gurobi, rls")
    parser.add_argument("--gurobi-time-limit", type=float, default=None, help="Gurobi time limit (seconds) per quasi-clique")
    parser.add_argument("--bnb-time-limit", type=float, default=None, help="BnB time limit (seconds) per quasi-clique")
    parser.add_argument("--bnb-node-limit", type=int, default=None, help="BnB max nodes explored per quasi-clique")
    parser.add_argument("--rls-max-steps", type=int, default=5000, help="RLS max steps per quasi-clique")
    parser.add_argument("--rls-seed", type=int, default=None, help="RLS random seed")
    
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
    method_kwargs = {}
    if args.method == "gurobi" and args.gurobi_time_limit is not None:
        method_kwargs["time_limit"] = args.gurobi_time_limit
    if args.method == "branch_and_bound":
        if args.bnb_time_limit is not None:
            method_kwargs["time_limit"] = args.bnb_time_limit
        if args.bnb_node_limit is not None:
            method_kwargs["node_limit"] = args.bnb_node_limit
    if args.method == "rls":
        method_kwargs["max_steps"] = args.rls_max_steps
        if args.rls_seed is not None:
            method_kwargs["seed"] = args.rls_seed
    print(f"\nPartitioning graph into K={args.K} subgraphs...")
    labels = quasi_clique_partition(
        adj_matrix, args.K, args.gamma, args.min_clique_size,
        method=args.method, **method_kwargs
    )
    
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
        "quasi_clique_algorithm": args.method,
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
