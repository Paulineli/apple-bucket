# Hypothesis Testing for Entity Binding

This project implements hypothesis testing for entity binding tasks using Boundless DAS (Distributed Alignment Search). It creates datasets with multiple entity groups, filters them based on model performance, and uses Boundless DAS to find alignment of the `positional_query_group` variable in language models.

## Overview

The project:
1. Creates a randomly sampled entity binding dataset with **10 groups** and **3 entities per group** (using the action template: person, object, location)
2. Filters the dataset to keep only examples where the model correctly predicts both the base input and counterfactual (>80% accuracy required)
3. Uses **Boundless DAS** to find alignment of `positional_query_group` across all layers at the last token position
4. Saves weights and IIA (Interchange Intervention Accuracy) for all tested alignments

## Requirements

- Python 3.8+
- PyTorch
- The `causalab` package (at the project root)
- A GPU (recommended, but can run on CPU)

## Usage

### Basic Usage

```bash
# Run with GPT-2 (default)
python run_hypothesis_test.py --model gpt2

# Specify GPU
python run_hypothesis_test.py --model gpt2 --gpu 0

# Custom dataset size
python run_hypothesis_test.py --model gpt2 --size 256

# Test mode (smaller dataset for quick testing)
python run_hypothesis_test.py --model gpt2 --test
```

### Command Line Arguments

- `--model`: HuggingFace model ID (default: `gpt2`)
- `--gpu`: GPU ID to use (default: use `cuda:0` if available, else `cpu`)
- `--size`: Dataset size to generate (default: 128)
- `--num-groups`: Number of entity groups (default: 10)
- `--output`: Output directory (default: auto-generated)
- `--batch-size`: Batch size for filtering and training (default: 32)
- `--n-features`: Number of features for DAS (default: 32)
- `--min-accuracy`: Minimum accuracy threshold (default: 0.8)
- `--test`: Test mode with reduced sizes

### Example: Running with Different Models

```bash
# GPT-2
python run_hypothesis_test.py --model gpt2 --gpu 0

# GPT-2 Medium
python run_hypothesis_test.py --model gpt2-medium --gpu 0

# Llama (if available)
python run_hypothesis_test.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 0
```

## Output Structure

The script creates the following output structure:

```
hypothesis_testing/outputs/{model_name}/
├── train_dataset/              # Filtered training dataset
├── test_dataset/               # Filtered test dataset
├── boundless_das/              # Boundless DAS results
│   ├── metadata.json
│   ├── models/                 # Trained models
│   ├── train_eval/             # Training set evaluation
│   ├── test_eval/              # Test set evaluation
│   └── heatmaps/               # Feature count heatmaps
│       └── positional_query_group_features.png
├── summary.json                # Summary of results
└── boundless_das_result.pkl    # Full result object
```

### Key Output Files

1. **summary.json**: Contains:
   - Model information
   - Dataset statistics
   - Filter accuracy
   - Best layer and scores
   - IIA accuracy for each layer

2. **boundless_das_result.pkl**: Full result object with:
   - `test_scores`: Dictionary mapping layer -> test score (IIA accuracy)
   - `train_scores`: Dictionary mapping layer -> train score
   - `feature_indices`: Selected feature indices for each layer
   - `metadata`: Complete experiment metadata

## Understanding Results

### IIA (Interchange Intervention Accuracy)

IIA accuracy measures how well the learned alignment can perform interchange interventions:
- **1.0**: Perfect alignment - the learned features can perfectly substitute the causal variable
- **0.0**: No alignment - the learned features cannot substitute the causal variable

### Best Layer

The script identifies the layer with the highest IIA accuracy for `positional_query_group`. This indicates where in the model this variable is most clearly represented.

### Feature Counts

The heatmap visualization shows how many features (out of `n_features`) are selected at each layer. More features may indicate:
- More complex representations
- Less precise alignment
- Need for more features to capture the variable

## Accuracy Threshold

The script requires >80% accuracy on the filtering task. If accuracy is below this threshold, it will:
1. Print a warning
2. Suggest using a larger model
3. Exit with an error

This ensures that only models capable of the task are used for alignment analysis.

## Dataset Details

### Entity Binding Task

The task uses the "action" template structure:
- **Statement template**: "{person} put {object} in the {location}"
- **Example**: "Pete put jam in the cup, Ann put water in the box, ..."

### Groups and Entities

- **10 groups**: Each group contains 3 entities (person, object, location)
- **3 entities per group**: Following the action template structure
- **Query**: The model is asked to retrieve information about a specific group

### Counterfactuals

The `swap_query_group` counterfactual swaps entity groups while keeping the query position the same, testing whether the model uses positional information.

## Troubleshooting

### Low Accuracy Warning

If you see:
```
⚠ WARNING: Task accuracy (XX%) is below threshold (80%)
```

**Solutions**:
- Use a larger model (e.g., `gpt2-medium` instead of `gpt2`)
- Check if the model is capable of the task
- Try increasing the dataset size

### Out of Memory

If you run out of memory:
- Reduce `--batch-size` (e.g., `--batch-size 16`)
- Use a smaller model
- Reduce `--n-features` (e.g., `--n-features 16`)
- Use `--test` mode for quick checks

### Import Errors

If you get import errors:
- Make sure you're in the correct directory
- Check that the `causalab` package is accessible (at project root)
- Scripts add the project-root causalab to sys.path automatically

## Code Structure

- `run_hypothesis_test.py`: Main script that orchestrates the entire pipeline
- `create_custom_config()`: Creates task configuration with 10 groups, 3 entities
- `create_dataset_with_n_groups()`: Generates dataset with specified number of groups
- `filter_dataset_with_accuracy_check()`: Filters dataset and checks accuracy threshold
- Boundless DAS training: Uses the `train_boundless_DAS` function from the causalab package

## References

- Entity Binding: Tests how models retrieve associated information
- Boundless DAS: Feature-level subspace alignment method
- Positional Query Group: Variable representing which group contains the queried entity

## License

See the main repository license.

