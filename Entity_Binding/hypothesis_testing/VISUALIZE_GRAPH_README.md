# Graph Visualization Tool

A comprehensive tool for visualizing graphs loaded from pickle files, with support for cluster analysis and multiple visualization types.

## Features

1. **Network Layout Visualization**: Multiple layout algorithms (spring, circular, kamada_kawai, spectral)
2. **Adjacency Matrix Heatmap**: Visual representation of the graph's connectivity
3. **Degree Distribution**: Histogram and log-log plots of node degrees
4. **Cluster Analysis**: If partition results are provided:
   - Cluster sizes
   - Intra-cluster vs inter-cluster edge densities
   - Query group distribution by cluster
   - Cluster IIA (Interchange Intervention Accuracy)

## Usage

### Basic Usage (Graph Only)

```bash
python3 visualize_graph.py --graph-path partition_results/graph_filling_liquids_23_10_512_Qwen_Qwen3-4B-Instruct-2507.pkl
```

### With Partition Results and Samples

```bash
python3 visualize_graph.py \
    --graph-path partition_results/graph_filling_liquids_23_10_512_Qwen_Qwen3-4B-Instruct-2507.pkl \
    --partition-results partition_results/partition_results_filling_liquids_23_10_512_Qwen_Qwen3-4B-Instruct-2507.json \
    --samples partition_results/filtered_input_samples_filling_liquids_23_10_512_Qwen_Qwen3-4B-Instruct-2507.pkl
```

### With Different Layout Algorithm

```bash
python3 visualize_graph.py \
    --graph-path partition_results/graph.pkl \
    --layout kamada_kawai
```

Available layouts:
- `spring` (default): Force-directed layout
- `circular`: Circular arrangement
- `kamada_kawai`: Kamada-Kawai force-directed layout
- `spectral`: Spectral layout based on graph Laplacian

### Collapsed Graph by Query Group

Collapse all nodes into 10 super-nodes based on their query_group, where:
- Each super-node represents all nodes with the same query_group (0-9)
- Edge weights between super-nodes = edge density (edges / possible edges)
- Each super-node has a self-loop showing internal edge density

```bash
python3 visualize_graph.py \
    --graph-path partition_results/graph.pkl \
    --samples partition_results/samples.pkl \
    --collapsed \
    --layout spring
```

This creates:
- `{graph_name}_collapsed_{layout}.png`: Network visualization with 10 query group nodes
- `{graph_name}_collapsed_heatmap.png`: Heatmap of the collapsed adjacency matrix

### Options

- `--graph-path`: **Required**. Path to the graph pickle file (adjacency matrix)
- `--partition-results`: Optional. Path to partition_results.json for cluster visualization
- `--samples`: Optional. Path to samples.pkl for query group analysis
- `--output-dir`: Optional. Output directory (default: same as graph directory)
- `--layout`: Layout algorithm (default: spring)
- `--no-heatmap`: Skip adjacency matrix heatmap (useful for very large graphs)
- `--collapsed`: Generate collapsed graph visualization (groups nodes by query_group)
- `--num-query-groups`: Number of query groups for collapsed mode (default: 10)

## Output Files

The script generates the following visualizations:

1. `{graph_name}_heatmap.png`: Adjacency matrix as a heatmap
2. `{graph_name}_network_{layout}.png`: Network graph visualization
3. `{graph_name}_degree_distribution.png`: Degree distribution plots
4. `{graph_name}_cluster_analysis.png`: Cluster analysis (if partition results provided)
5. `{graph_name}_collapsed_{layout}.png`: Collapsed graph by query group (if --collapsed used)
6. `{graph_name}_collapsed_heatmap.png`: Collapsed adjacency matrix heatmap (if --collapsed used)

## Graph Statistics

The script prints comprehensive statistics including:
- Number of nodes and edges
- Edge density
- Average, max, and min degree
- Number of connected components
- Average clustering coefficient
- Partition information (if available)

## Example Output

```
GRAPH STATISTICS
============================================================
Number of nodes: 509
Number of edges: 27405
Edge density: 0.2120
Average degree: 107.68
Max degree: 260
Min degree: 13
Number of connected components: 1
Largest component size: 509
Average clustering coefficient: 0.5163
```

## Tips

1. **Large Graphs**: For graphs with >1000 nodes, use `--no-heatmap` to skip the adjacency matrix visualization
2. **Layout Selection**: 
   - `spring` works well for most graphs
   - `kamada_kawai` is better for smaller, well-connected graphs
   - `circular` is useful for seeing overall structure
3. **Cluster Visualization**: Provide partition results to see how clusters are distributed in the network
4. **Query Group Analysis**: Provide samples to see query group distribution across clusters
5. **Collapsed Mode**: Use `--collapsed` with `--samples` to see a simplified view with 10 query group nodes. This is especially useful for understanding connectivity patterns between different query group positions.


# How to Read the Network Graph Visualization

## Understanding the Network Graph

The network graph shows the structure of your graph where:
- **Nodes** (circles) represent individual samples/data points
- **Edges** (lines) represent connections between samples
- An edge exists between two nodes if they have consistent interchange interventions (both directions work correctly)

## Node Colors

### When Partition Results Are Provided (Your Case)

When you provide `--partition-results`, nodes are **colored by cluster**:

- **Different colors = Different clusters**
- The color scheme uses the `tab20` colormap which provides 20 distinct colors
- **Cluster 0** typically appears as **blue** (first color in tab20)
- **Cluster 1** typically appears as **orange/red** (second color in tab20)
- **Cluster 2** would be **green**, and so on...

**In your specific case:**
- You have **2 clusters** (K=2)
- **Cluster 0**: 146 nodes (IIA = 0.9014) - likely appears as **blue**
- **Cluster 1**: 363 nodes (IIA = 0.0759) - likely appears as **orange/red**

### When Partition Results Are NOT Provided

If you don't provide partition results, **all nodes are light blue** - this means no cluster information is being displayed.

## Layout Algorithms Explained

### Spring Layout
- **Force-directed layout**: Nodes repel each other, edges act like springs
- **Dense clusters** appear as tightly packed groups
- **Sparse regions** have nodes spread apart
- **Interpretation**: 
  - Nodes close together = more connections between them
  - Well-separated groups = distinct communities/clusters

### Circular Layout
- **All nodes arranged in a circle**
- **Uniform spacing** regardless of connections
- **Useful for**: Seeing overall structure and identifying clusters by color
- **Interpretation**: 
  - Look for color groupings around the circle
  - Dense edge connections appear as thick "spokes" or "chords"

### Kamada-Kawai Layout
- **Another force-directed algorithm**
- Tries to minimize edge crossings
- **Better for smaller graphs** (< 200 nodes)
- **Interpretation**: Similar to spring, but often more structured

### Spectral Layout
- **Based on graph Laplacian eigenvectors**
- Positions nodes using spectral properties
- **Useful for**: Identifying community structure

## Reading Your Graph

### What to Look For:

1. **Color Clustering**:
   - In spring layout, nodes of the same color should cluster together
   - This indicates that samples in the same cluster are well-connected
   - If colors are mixed, the clustering may not be optimal

2. **Edge Density**:
   - **Dense regions** (many edges) = high connectivity
   - **Sparse regions** (few edges) = low connectivity
   - Your graph has edge density of 0.2120 (21.2% of possible edges exist)

3. **Cluster Separation**:
   - **Well-separated clusters** = good partitioning
   - **Overlapping clusters** = samples from different clusters are still connected
   - Your Cluster 0 has IIA=0.9014 (very dense), Cluster 1 has IIA=0.0759 (sparse)

4. **Node Labels**:
   - For graphs ≤100 nodes: All nodes are labeled with their index
   - For larger graphs: Only high-degree nodes (top 10%) are labeled
   - Your graph has 509 nodes, so only high-degree nodes are labeled

## Interpreting Your Specific Graph

Based on your partition results:

```
Cluster 0: 146 nodes, IIA = 0.9014 (very dense - 90% of possible edges exist)
Cluster 1: 363 nodes, IIA = 0.0759 (sparse - only 7.6% of possible edges exist)
```

**What this means:**
- **Cluster 0** (blue nodes): A tightly-knit group where almost all samples can interchange with each other
- **Cluster 1** (orange/red nodes): A looser group with fewer interconnections
- The **large difference in IIA** suggests these are two distinct behavioral patterns

**In the visualization:**
- **Spring layout**: You should see Cluster 0 as a dense blue cluster, Cluster 1 as a more spread-out orange/red region
- **Circular layout**: You should see blue nodes grouped together and orange/red nodes grouped together around the circle

## Tips for Better Visualization

1. **Check the legend**: If clusters are shown, there's a legend in the upper right
2. **Compare layouts**: Try different layouts to see the structure from different perspectives
3. **Look at cluster analysis plot**: The `_cluster_analysis.png` file shows detailed statistics
4. **Check degree distribution**: High-degree nodes are "hubs" that connect to many other nodes

## Common Questions

**Q: Why are all nodes blue?**
A: You didn't provide `--partition-results`, so no cluster coloring is applied.

**Q: Why do I see mixed colors in spring layout?**
A: This could mean:
- The clusters are not well-separated (samples from different clusters are still connected)
- The layout algorithm is still optimizing positions
- Try circular layout to see if colors group better

**Q: What do the gray lines mean?**
A: Gray lines are edges - they show which samples are connected (can interchange successfully).

**Q: Why are some nodes labeled and others not?**
A: For large graphs (>100 nodes), only high-degree nodes (top 10%) are labeled to reduce clutter.
