#!/usr/bin/env python
"""
Plot Query Group Distribution for Each Subgraph

This script reads partition results and plots the distribution of query_group
values for samples within each subgraph.
"""

import json
import pickle
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_partition_results(results_dir: Path):
    """Load partition results and samples."""
    results_path = results_dir / "partition_results_filling_liquids_15_10_512_google_gemma-2-2b-it.json"
    samples_path = results_dir / "filtered_input_samples_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl"
    
    # Load partition results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load samples
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    
    return results, samples


def compute_query_group_distributions(results: dict, samples: list):
    """Compute query_group distribution for each subgraph."""
    labels = results['labels']
    K = results['K']
    
    # Group samples by subgraph
    subgraph_samples = defaultdict(list)
    for idx, label in enumerate(labels):
        if idx < len(samples):
            subgraph_samples[label].append(samples[idx])
    
    # Compute query_group distribution for each subgraph
    distributions = {}
    for k in range(K):
        if k in subgraph_samples:
            query_groups = [sample['query_group'] for sample in subgraph_samples[k]]
            counter = Counter(query_groups)
            total = len(query_groups)
            distributions[k] = {
                'counts': dict(counter),
                'proportions': {qg: count / total for qg, count in counter.items()},
                'total': total
            }
        else:
            distributions[k] = {
                'counts': {},
                'proportions': {},
                'total': 0
            }
    
    return distributions


def plot_distributions(distributions: dict, output_path: Path, results: dict):
    """Plot query_group distributions for each subgraph."""
    K = len(distributions)
    
    # Determine number of rows and columns for subplots
    n_cols = min(3, K)
    n_rows = (K + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if K == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Find all unique query_group values across all subgraphs
    all_query_groups = set()
    for dist in distributions.values():
        all_query_groups.update(dist['counts'].keys())
    all_query_groups = sorted(all_query_groups)
    
    # Plot each subgraph
    for k in range(K):
        ax = axes[k]
        dist = distributions[k]
        
        if dist['total'] == 0:
            ax.text(0.5, 0.5, f'Subgraph {k}\n(No samples)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Subgraph {k} (Empty)')
            continue
        
        # Get counts for all query_groups (0 for missing ones)
        counts = [dist['counts'].get(qg, 0) for qg in all_query_groups]
        proportions = [dist['proportions'].get(qg, 0.0) for qg in all_query_groups]
        
        # Create bar plot
        x_pos = np.arange(len(all_query_groups))
        bars = ax.bar(x_pos, proportions, alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Query Group', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title(f'Subgraph {k} (n={dist["total"]}) density: {results["iia_by_cluster"][str(k)]:.2f}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_query_groups)
        ax.set_ylim(0, max(proportions) * 1.15 if proportions else 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for k in range(K, len(axes)):
        axes[k].axis('off')
    
    # Add overall title
    task = results.get('task', 'unknown')
    model = results.get('model', 'unknown')
    layer = results.get('layer', 'unknown')
    fig.suptitle(f'Query Group Distribution by Subgraph\n{task} | {model} | Layer {layer}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    return fig


def print_summary(distributions: dict, results: dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Query Group Distribution Summary")
    print("="*60)
    
    K = len(distributions)
    for k in range(K):
        dist = distributions[k]
        print(f"\nSubgraph {k} (n={dist['total']}):")
        if dist['total'] == 0:
            print("  No samples")
            continue
        
        # Sort by query_group
        sorted_items = sorted(dist['counts'].items())
        for qg, count in sorted_items:
            prop = dist['proportions'][qg]
            print(f"  Query Group {qg}: {count} ({prop*100:.1f}%)")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot query group distribution for each subgraph"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./partition_results",
        help="Directory containing partition_results.json and filtered_input_samples.pkl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: results_dir/query_group_distribution.png)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print summary, don't create plot"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Load data
    print(f"Loading partition results from {results_dir}...")
    results, samples = load_partition_results(results_dir)
    print(f"Loaded {len(samples)} samples with {results['K']} subgraphs")
    
    # Compute distributions
    distributions = compute_query_group_distributions(results, samples)
    
    # Print summary
    print_summary(distributions, results)
    
    # Create plot
    if not args.no_plot:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = results_dir / "query_group_distribution.png"
        
        plot_distributions(distributions, output_path, results)
        print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
