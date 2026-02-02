#!/usr/bin/env python
"""
Comprehensive Graph Visualization Tool

This script provides multiple visualizations for graphs loaded from pickle files:
1. Network layout visualization (multiple layouts)
2. Adjacency matrix heatmap
3. Graph statistics and metrics
4. Cluster visualization (if partition results available)
5. Degree distribution

Usage:
    python visualize_graph.py --graph-path partition_results/graph_filling_liquids_23_10_512_Qwen_Qwen3-4B-Instruct-2507.pkl
    python visualize_graph.py --graph-path graph.pkl --partition-results partition_results.json --samples samples.pkl
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter, defaultdict


def load_graph(graph_path: Path) -> np.ndarray:
    """Load adjacency matrix from pickle file."""
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        adj_matrix = pickle.load(f)
    
    # Ensure it's a numpy array
    if not isinstance(adj_matrix, np.ndarray):
        adj_matrix = np.array(adj_matrix)
    
    # Ensure symmetric (undirected graph)
    if not np.array_equal(adj_matrix, adj_matrix.T):
        print("  Warning: Graph is not symmetric. Making it symmetric...")
        adj_matrix = adj_matrix | adj_matrix.T
    
    print(f"  Graph loaded: {adj_matrix.shape[0]} nodes, {np.sum(adj_matrix) // 2} edges")
    return adj_matrix


def load_partition_results(partition_path: Optional[Path]) -> Optional[Dict]:
    """Load partition results if available."""
    if partition_path is None or not partition_path.exists():
        return None
    
    print(f"Loading partition results from {partition_path}...")
    with open(partition_path, 'r') as f:
        results = json.load(f)
    
    labels = np.array(results.get('labels', []))
    print(f"  Loaded partition with {len(np.unique(labels))} clusters")
    return results


def load_samples(samples_path: Optional[Path]) -> Optional[List]:
    """Load samples if available."""
    if samples_path is None or not samples_path.exists():
        return None
    
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"  Loaded {len(samples)} samples")
    return samples


def compute_graph_statistics(adj_matrix: np.ndarray) -> Dict:
    """Compute various graph statistics."""
    n = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix) // 2
    max_edges = n * (n - 1) // 2
    edge_density = num_edges / max_edges if max_edges > 0 else 0.0
    
    # Build NetworkX graph for additional metrics
    G = nx.from_numpy_array(adj_matrix)
    
    # Compute degrees
    degrees = [G.degree(i) for i in range(n)]
    avg_degree = np.mean(degrees)
    
    # Connected components
    num_components = nx.number_connected_components(G)
    largest_component_size = len(max(nx.connected_components(G), key=len)) if num_components > 0 else 0
    
    # Clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0.0
    
    stats = {
        'num_nodes': n,
        'num_edges': num_edges,
        'edge_density': edge_density,
        'avg_degree': avg_degree,
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0,
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'avg_clustering': avg_clustering,
    }
    
    return stats, G, degrees


def plot_adjacency_heatmap(adj_matrix: np.ndarray, output_path: Path, title: str = "Adjacency Matrix"):
    """Plot adjacency matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(adj_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge (1=connected, 0=disconnected)', fontsize=11)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, adj_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, adj_matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved adjacency heatmap to {output_path}")
    plt.close()


def plot_network_layout(
    G: nx.Graph,
    adj_matrix: np.ndarray,
    output_path: Path,
    partition_results: Optional[Dict] = None,
    samples: Optional[List] = None,
    layout_type: str = 'spring'
):
    """Plot network graph with different layout algorithms."""
    n = adj_matrix.shape[0]
    
    # Choose layout algorithm
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=1/np.sqrt(n), iterations=50, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'spectral':
        try:
            pos = nx.spectral_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Color nodes by cluster if partition results available
    node_colors = None
    cluster_labels = None
    if partition_results is not None:
        labels = np.array(partition_results.get('labels', []))
        if len(labels) == n:
            unique_labels = np.unique(labels)
            # Create color map
            cmap = plt.colormaps.get_cmap('tab20')
            node_colors = [cmap(np.where(unique_labels == labels[i])[0][0] / len(unique_labels)) for i in range(n)]
            cluster_labels = labels
    else:
        node_colors = 'lightblue'
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax, edge_color='gray')
    
    # Draw nodes
    if node_colors == 'lightblue':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax, alpha=0.8)
        legend_text = "All nodes (no cluster information)"
    else:
        # Draw nodes by cluster
        cluster_info = {}
        for cluster_id in np.unique(cluster_labels):
            nodes_in_cluster = [i for i in range(n) if cluster_labels[i] == cluster_id]
            cluster_color = node_colors[nodes_in_cluster[0]]
            cluster_size = len(nodes_in_cluster)
            
            # Get IIA if available
            iia_info = ""
            if partition_results and 'iia_by_cluster' in partition_results:
                iia_val = partition_results['iia_by_cluster'].get(str(cluster_id), None)
                if iia_val is not None:
                    iia_info = f" (IIA={iia_val:.3f}, n={cluster_size})"
                else:
                    iia_info = f" (n={cluster_size})"
            else:
                iia_info = f" (n={cluster_size})"
            
            cluster_info[cluster_id] = {
                'color': cluster_color,
                'size': cluster_size,
                'iia': iia_info
            }
            
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes_in_cluster,
                node_color=[cluster_color],
                node_size=50,
                ax=ax,
                alpha=0.8,
                label=f'Cluster {cluster_id}{iia_info}'
            )
        legend_text = "Nodes colored by cluster"
    
    # Draw labels (only for small graphs or high-degree nodes)
    if n <= 100:
        labels_dict = {i: str(i) for i in range(n)}
        nx.draw_networkx_labels(G, pos, labels_dict, font_size=6, ax=ax)
    else:
        # Only label high-degree nodes
        degrees = dict(G.degree())
        high_degree_nodes = [i for i in range(n) if degrees[i] >= np.percentile(list(degrees.values()), 90)]
        labels_dict = {i: str(i) for i in high_degree_nodes}
        nx.draw_networkx_labels(G, pos, labels_dict, font_size=6, ax=ax)
    
    # Create informative title
    title = f'Network Graph ({layout_type} layout)'
    if partition_results is not None and node_colors != 'lightblue':
        title += f'\n{legend_text}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend if clusters are shown
    if partition_results is not None and node_colors != 'lightblue':
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, title='Clusters')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved network layout ({layout_type}) to {output_path}")
    plt.close()


def plot_degree_distribution(degrees: List[int], output_path: Path):
    """Plot degree distribution histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(degrees, bins=min(50, max(10, len(set(degrees)))), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Degree Distribution (Histogram)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Log-log plot (for scale-free networks)
    degree_counts = Counter(degrees)
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_sorted]
    
    ax2.loglog(degrees_sorted, counts, 'bo', markersize=6, alpha=0.7)
    ax2.set_xlabel('Degree (log scale)', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_title('Degree Distribution (Log-Log)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved degree distribution to {output_path}")
    plt.close()


def create_collapsed_graph_by_query_group(
    adj_matrix: np.ndarray,
    samples: List[Dict],
    num_groups: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Collapse graph into super-nodes based on query_group.
    
    Args:
        adj_matrix: Original adjacency matrix
        samples: List of sample dictionaries with 'query_group' key
        num_groups: Number of query groups (default: 10)
    
    Returns:
        Tuple of (collapsed_adj_matrix, metadata_dict)
        collapsed_adj_matrix[i, j] = edge density between query_group i and j
        metadata contains node counts per group
    """
    n = adj_matrix.shape[0]
    if len(samples) != n:
        raise ValueError(f"Samples length ({len(samples)}) doesn't match graph size ({n})")
    
    # Group nodes by query_group
    nodes_by_group = defaultdict(list)
    for idx, sample in enumerate(samples):
        query_group = sample.get('query_group', -1)
        if 0 <= query_group < num_groups:
            nodes_by_group[query_group].append(idx)
        else:
            # Handle out-of-range query groups
            print(f"  Warning: Sample {idx} has query_group={query_group}, which is out of range [0, {num_groups-1}]")
    
    # Create collapsed adjacency matrix
    collapsed_adj = np.zeros((num_groups, num_groups), dtype=float)
    group_sizes = {}
    
    for i in range(num_groups):
        nodes_i = nodes_by_group.get(i, [])
        group_sizes[i] = len(nodes_i)
        
        if len(nodes_i) == 0:
            continue
        
        # Self-loop: edges within group i
        if len(nodes_i) > 1:
            # Count edges within group i
            edges_within = 0
            for idx1 in nodes_i:
                for idx2 in nodes_i:
                    if idx1 < idx2 and adj_matrix[idx1, idx2]:
                        edges_within += 1
            
            # Maximum possible edges within group i
            max_edges_within = len(nodes_i) * (len(nodes_i) - 1) // 2
            if max_edges_within > 0:
                collapsed_adj[i, i] = edges_within / max_edges_within
            else:
                collapsed_adj[i, i] = 0.0
        
        # Edges between group i and group j (j > i)
        for j in range(i + 1, num_groups):
            nodes_j = nodes_by_group.get(j, [])
            if len(nodes_j) == 0:
                continue
            
            # Count edges between groups i and j
            edges_between = 0
            for idx_i in nodes_i:
                for idx_j in nodes_j:
                    if adj_matrix[idx_i, idx_j]:
                        edges_between += 1
            
            # Maximum possible edges between groups i and j
            max_edges_between = len(nodes_i) * len(nodes_j)
            if max_edges_between > 0:
                edge_density = edges_between / max_edges_between
                collapsed_adj[i, j] = edge_density
                collapsed_adj[j, i] = edge_density  # Make symmetric
            else:
                collapsed_adj[i, j] = 0.0
                collapsed_adj[j, i] = 0.0
    
    metadata = {
        'group_sizes': group_sizes,
        'nodes_by_group': {k: len(v) for k, v in nodes_by_group.items()}
    }
    
    return collapsed_adj, metadata


def plot_collapsed_graph(
    collapsed_adj: np.ndarray,
    metadata: Dict,
    output_path: Path,
    layout_type: str = 'spring'
):
    """
    Plot collapsed graph where nodes represent query groups.
    
    Args:
        collapsed_adj: Collapsed adjacency matrix (edge densities)
        metadata: Metadata with group_sizes
        output_path: Output file path
        layout_type: Layout algorithm
    """
    num_groups = collapsed_adj.shape[0]
    group_sizes = metadata.get('group_sizes', {})
    
    # Create NetworkX graph (use Graph for regular edges, track self-loops separately)
    G = nx.Graph()
    self_loops = {}  # Track self-loops separately: {node: weight}
    
    # Add nodes with sizes proportional to group size
    for i in range(num_groups):
        size = group_sizes.get(i, 0)
        G.add_node(i, size=size)
    
    # Add edges with weights
    for i in range(num_groups):
        for j in range(i, num_groups):  # Include self-loops (i == j)
            weight = collapsed_adj[i, j]
            if weight > 0:
                if i == j:
                    # Self-loop - store separately
                    self_loops[i] = weight
                else:
                    # Edge between different groups
                    G.add_edge(i, j, weight=weight)
    
    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'spectral':
        try:
            pos = nx.spectral_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw edges between different nodes
    edges_between = list(G.edges())
    if edges_between:
        edge_weights = [G[u][v]['weight'] for u, v in edges_between]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [w * 3 / max_weight for w in edge_weights]  # Scale for visibility
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges_between,
            width=edge_widths,
            alpha=0.6,
            edge_color='gray',
            ax=ax
        )
    
    # Draw self-loops
    for node, weight in self_loops.items():
        if node in pos:
            # Draw self-loop as a curved arc
            x, y = pos[node]
            # Create a small circle around the node
            circle = plt.Circle((x, y), 0.15, fill=False, edgecolor='blue', 
                              linewidth=weight * 5, 
                              alpha=0.7)
            ax.add_patch(circle)
    
    # Draw nodes with sizes proportional to group size
    node_sizes = [group_sizes.get(i, 0) * 100 for i in range(num_groups)]  # Scale for visibility
    node_colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, num_groups))
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # Draw labels
    labels = {i: f'QG{i}\n(n={group_sizes.get(i, 0)})' for i in range(num_groups)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
    
    # Add edge weight labels for significant edges
    for u, v in edges_between:
        weight = G[u][v]['weight']
        if weight > 0.1:  # Only label significant edges
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add self-loop labels
    for node, weight in self_loops.items():
        if node in pos:
            x, y = pos[node]
            ax.text(x, y + 0.25, f'self={weight:.2f}',
                   fontsize=8, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax.set_title(f'Collapsed Graph by Query Group ({layout_type} layout)\n'
                f'Node size = number of samples, Edge width = edge density',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved collapsed graph to {output_path}")
    plt.close()


def plot_cluster_analysis(
    adj_matrix: np.ndarray,
    partition_results: Dict,
    samples: Optional[List],
    output_path: Path
):
    """Plot cluster analysis including inter-cluster and intra-cluster edge densities."""
    labels = np.array(partition_results.get('labels', []))
    n = adj_matrix.shape[0]
    
    if len(labels) != n:
        print(f"  Warning: Partition labels length ({len(labels)}) doesn't match graph size ({n})")
        return
    
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    
    # Compute intra-cluster and inter-cluster edge densities
    intra_cluster_edges = 0
    inter_cluster_edges = 0
    intra_cluster_max = 0
    inter_cluster_max = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                if labels[i] == labels[j]:
                    intra_cluster_edges += 1
                else:
                    inter_cluster_edges += 1
    
    # Compute max possible edges
    for k in unique_labels:
        cluster_size = np.sum(labels == k)
        intra_cluster_max += cluster_size * (cluster_size - 1) // 2
    
    inter_cluster_max = n * (n - 1) // 2 - intra_cluster_max
    
    intra_density = intra_cluster_edges / intra_cluster_max if intra_cluster_max > 0 else 0.0
    inter_density = inter_cluster_edges / inter_cluster_max if inter_cluster_max > 0 else 0.0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Cluster sizes
    ax = axes[0, 0]
    cluster_sizes = [np.sum(labels == k) for k in unique_labels]
    ax.bar(range(K), cluster_sizes, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cluster ID', fontsize=11)
    ax.set_ylabel('Number of Nodes', fontsize=11)
    ax.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'C{k}' for k in unique_labels])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Intra vs Inter cluster densities
    ax = axes[0, 1]
    categories = ['Intra-Cluster', 'Inter-Cluster']
    densities = [intra_density, inter_density]
    colors = ['green', 'red']
    bars = ax.bar(categories, densities, alpha=0.7, color=colors, edgecolor='black')
    ax.set_ylabel('Edge Density', fontsize=11)
    ax.set_title('Intra-Cluster vs Inter-Cluster Edge Density', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(densities) * 1.2 if densities else 1.0)
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{density:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Query group distribution by cluster (if samples available)
    ax = axes[1, 0]
    if samples is not None and len(samples) == n:
        query_group_dist = defaultdict(lambda: defaultdict(int))
        for idx, sample in enumerate(samples):
            cluster_id = labels[idx]
            query_group = sample.get('query_group', -1)
            query_group_dist[cluster_id][query_group] += 1
        
        # Create stacked bar chart
        all_query_groups = sorted(set(qg for dist in query_group_dist.values() for qg in dist.keys()))
        bottom = np.zeros(K)
        colors_qg = plt.colormaps.get_cmap('tab10')
        
        for qg_idx, qg in enumerate(all_query_groups):
            counts = [query_group_dist[k].get(qg, 0) for k in unique_labels]
            ax.bar(range(K), counts, bottom=bottom, label=f'QG {qg}',
                  color=colors_qg(qg_idx / len(all_query_groups)), alpha=0.7, edgecolor='black')
            bottom += counts
        
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Number of Nodes', fontsize=11)
        ax.set_title('Query Group Distribution by Cluster', fontsize=12, fontweight='bold')
        ax.set_xticks(range(K))
        ax.set_xticklabels([f'C{k}' for k in unique_labels])
        ax.legend(title='Query Group', fontsize=8, title_fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Samples not available\nfor query group analysis',
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Query Group Distribution by Cluster', fontsize=12, fontweight='bold')
    
    # 4. Cluster IIA (from partition results if available)
    ax = axes[1, 1]
    if 'iia_by_cluster' in partition_results:
        iia_by_cluster = partition_results['iia_by_cluster']
        cluster_ids = [int(k) for k in sorted(iia_by_cluster.keys())]
        iia_values = [iia_by_cluster[str(k)] for k in cluster_ids]
        
        bars = ax.bar(range(len(cluster_ids)), iia_values, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('IIA (Edge Density)', fontsize=11)
        ax.set_title('Cluster IIA', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f'C{k}' for k in cluster_ids])
        ax.set_ylim(0, max(iia_values) * 1.2 if iia_values else 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, iia in zip(bars, iia_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{iia:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'IIA by cluster not available\nin partition results',
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Cluster IIA', fontsize=12, fontweight='bold')
    
    plt.suptitle('Cluster Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved cluster analysis to {output_path}")
    plt.close()


def print_statistics(stats: Dict, partition_results: Optional[Dict] = None):
    """Print graph statistics."""
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Edge density: {stats['edge_density']:.4f}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Max degree: {stats['max_degree']}")
    print(f"Min degree: {stats['min_degree']}")
    print(f"Number of connected components: {stats['num_components']}")
    print(f"Largest component size: {stats['largest_component_size']}")
    print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    
    if partition_results:
        print("\n" + "-"*60)
        print("PARTITION INFORMATION")
        print("-"*60)
        labels = np.array(partition_results.get('labels', []))
        unique_labels = np.unique(labels)
        print(f"Number of clusters: {len(unique_labels)}")
        print(f"K parameter: {partition_results.get('K', 'N/A')}")
        print(f"Gamma parameter: {partition_results.get('gamma', 'N/A')}")
        if 'overall_iia' in partition_results:
            print(f"Overall IIA: {partition_results['overall_iia']:.4f}")
        if 'iia_by_cluster' in partition_results:
            print("\nIIA by cluster:")
            for k in sorted(partition_results['iia_by_cluster'].keys()):
                iia = partition_results['iia_by_cluster'][k]
                cluster_size = np.sum(labels == int(k))
                print(f"  Cluster {k}: IIA = {iia:.4f} (size: {cluster_size})")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize graphs from pickle files with multiple views"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        required=True,
        help="Path to graph.pkl file (adjacency matrix)"
    )
    parser.add_argument(
        "--partition-results",
        type=str,
        default=None,
        help="Path to partition_results.json (optional, for cluster visualization)"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Path to samples.pkl (optional, for query group analysis)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: same as graph directory)"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "spectral"],
        help="Network layout algorithm (default: spring)"
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip adjacency matrix heatmap (useful for large graphs)"
    )
    parser.add_argument(
        "--collapsed",
        action="store_true",
        help="Generate collapsed graph visualization (groups nodes by query_group into 10 super-nodes)"
    )
    parser.add_argument(
        "--num-query-groups",
        type=int,
        default=10,
        help="Number of query groups for collapsed visualization (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = graph_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adj_matrix = load_graph(graph_path)
    partition_results = load_partition_results(
        Path(args.partition_results) if args.partition_results else None
    )
    samples = load_samples(Path(args.samples) if args.samples else None)
    
    # Compute statistics
    stats, G, degrees = compute_graph_statistics(adj_matrix)
    print_statistics(stats, partition_results)
    
    # Generate base filename
    graph_name = graph_path.stem
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Adjacency matrix heatmap
    if not args.no_heatmap:
        heatmap_path = output_dir / f"{graph_name}_heatmap.png"
        plot_adjacency_heatmap(adj_matrix, heatmap_path, f"Adjacency Matrix - {graph_name}")
    
    # 2. Network layout
    layout_path = output_dir / f"{graph_name}_network_{args.layout}.png"
    plot_network_layout(G, adj_matrix, layout_path, partition_results, samples, args.layout)
    
    # 3. Degree distribution
    degree_path = output_dir / f"{graph_name}_degree_distribution.png"
    plot_degree_distribution(degrees, degree_path)
    
    # 4. Cluster analysis (if partition results available)
    if partition_results is not None:
        cluster_path = output_dir / f"{graph_name}_cluster_analysis.png"
        plot_cluster_analysis(adj_matrix, partition_results, samples, cluster_path)
    
    # 5. Collapsed graph by query group (if requested and samples available)
    collapsed_path = None
    collapsed_heatmap_path = None
    if args.collapsed:
        if samples is None:
            print("  Warning: --collapsed requires --samples. Skipping collapsed visualization.")
        else:
            try:
                collapsed_adj, collapsed_metadata = create_collapsed_graph_by_query_group(
                    adj_matrix, samples, args.num_query_groups
                )
                collapsed_path = output_dir / f"{graph_name}_collapsed_{args.layout}.png"
                plot_collapsed_graph(collapsed_adj, collapsed_metadata, collapsed_path, args.layout)
                
                # Also create a heatmap of the collapsed adjacency matrix
                collapsed_heatmap_path = output_dir / f"{graph_name}_collapsed_heatmap.png"
                plot_adjacency_heatmap(
                    collapsed_adj, 
                    collapsed_heatmap_path, 
                    f"Collapsed Adjacency Matrix by Query Group - {graph_name}"
                )
            except Exception as e:
                print(f"  Error creating collapsed visualization: {e}")
    
    print(f"\n✓ All visualizations saved to {output_dir}")
    print(f"  - Network layout: {layout_path.name}")
    if not args.no_heatmap:
        print(f"  - Adjacency heatmap: {heatmap_path.name}")
    print(f"  - Degree distribution: {degree_path.name}")
    if partition_results is not None:
        print(f"  - Cluster analysis: {cluster_path.name}")
    if collapsed_path is not None:
        print(f"  - Collapsed graph: {collapsed_path.name}")
        print(f"  - Collapsed heatmap: {collapsed_heatmap_path.name}")


if __name__ == "__main__":
    main()
