#!/usr/bin/env python
"""
DAS Testing for Entity Binding

This script:
1. Creates a randomly sampled entity binding dataset with 10 groups and 3 entities
2. Filters the dataset based on model predictions (base and counterfactual must both be correct)
3. Checks task accuracy (must be >80%, otherwise suggests larger model)
4. Runs vanilla interchange interventions to find alignment of positional_query_group
5. Saves IIA accuracy for all layers tested

Usage:
    python run_das.py --model MODEL_ID [--gpu GPU_ID] [--size DATASET_SIZE]
    python run_das.py --model gpt2 --gpu 0 --size 1024
    python run_das.py --model Qwen/Qwen3-8B --hf-cache-dir /path/to/.cache --gpu 0
    python run_das.py --model MODEL_ID --task-type filling_liquids

Task types:
    - action: Person put object in location (default)
    - filling_liquids: Person fills container with liquid
"""

import sys
import os

# Set HuggingFace cache directory BEFORE importing transformers
# This must happen before any HuggingFace library is imported
for i, arg in enumerate(sys.argv):
    if arg == "--hf-cache-dir" and i + 1 < len(sys.argv):
        os.environ["HF_HOME"] = sys.argv[i + 1]
        print(f"Using HuggingFace cache directory: {sys.argv[i + 1]}")
        break
import argparse
import torch
import json
import pickle
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, Callable

# Add the causalab package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'causalab'))

from causalab.tasks.entity_binding.config import (
    EntityBindingTaskConfig, 
    create_sample_action_config,
    create_filling_liquids_config
)
from causalab.tasks.entity_binding.causal_models import create_positional_causal_model, create_direct_causal_model, sample_valid_entity_binding_input
from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.tasks.entity_binding.templates import TemplateProcessor
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.experiments.metric import causal_score_intervention_outputs
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import numpy as np


# ========== Register Qwen3 support for pyvene ==========
# Qwen3 has the same architecture as Qwen2, so we reuse the same mappings
def register_qwen3_for_pyvene():
    """Register Qwen3 model types with pyvene's type mappings."""
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
            qwen2_classifier_type_to_module_mapping,
            qwen2_classifier_type_to_dimension_mapping,
        )
        
        # Register Qwen3 models using Qwen2 mappings (same architecture)
        if hasattr(qwen3_modeling, 'Qwen3Model'):
            type_to_module_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForSequenceClassification'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_dimension_mapping
        
        print("Successfully registered Qwen3 support for pyvene")
    except ImportError as e:
        print(f"Warning: Could not register Qwen3 for pyvene: {e}")
    except Exception as e:
        print(f"Warning: Error registering Qwen3 for pyvene: {e}")

register_qwen3_for_pyvene()
# ========================================================

def create_custom_config(task_type: str = "action") -> EntityBindingTaskConfig:
    """
    Create a custom entity binding config with 10 groups.
    
    The number of entities per group depends on the task type:
    - action: 3 entities per group (person, object, location)
    - filling_liquids: 3 entities per group (person, container, liquid)
    
    Args:
        task_type: One of "action" (person put object in location) or 
                  "filling_liquids" (person fills container with liquid)
    
    Returns:
        EntityBindingTaskConfig configured for the specified task type
    """
    if task_type == "filling_liquids":
        config = create_filling_liquids_config()
        # Expand entity pools for filling_liquids task to support 10 groups with unique entities
        config.entity_pools[0] = [
            "John", "Mary", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
            "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
            "Ben", "Liz", "Tom", "Joy"
        ]
        config.entity_pools[1] = [
            "cup", "glass", "bottle", "mug", "jar", "pitcher", "bowl", "flask",
            "tumbler", "chalice", "vessel", "container", "tank", "can", "tube", "vial",
            "goblet", "stein", "carafe", "decanter"
        ]
        config.entity_pools[2] = [
            "beer", "wine", "water", "juice", "milk", "coffee", "tea", "soda",
            "lemonade", "smoothie", "soup", "broth", "sauce", "syrup", "oil", "honey",
            "cider", "nectar", "punch", "tonic"
        ]
    else:
        # Default to action task
        config = create_sample_action_config()
        
        # Expand entity pools to support 10 groups
        # Person pool
        config.entity_pools[0] = [
            "Pete", "Ann", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
            "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
            "Ben", "Liz", "Tom", "Joy"
        ]
        
        # Object pool
        config.entity_pools[1] = [
            "jam", "water", "book", "coin", "pen", "key", "phone", "watch",
            "cup", "box", "bag", "hat", "map", "card", "lamp", "ball",
            "rope", "tape", "tool", "clip"
        ]
        
        # Location pool
        config.entity_pools[2] = [
            "cup", "box", "table", "shelf", "drawer", "bag", "pocket", "basket",
            "desk", "chair", "floor", "rack", "case", "tray", "bin", "stand",
            "cabinet", "corner", "bench", "counter"
        ]
    
    # Set max_groups for all tasks
    config.max_groups = 10
    
    # IMPORTANT: Do NOT override max_entities_per_group here!
    # Each task type has its own correct value set in their configs:
    # - action: 3 (person, object, location)
    # - filling_liquids: 3 (person, container, liquid)
    
    # Add instruction wrapper for better performance
    config.prompt_prefix = "We will ask a question about the following sentences. Only return the answer, no other text.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"
    config.fixed_query_indices = (0,)
    config.fixed_answer_index = 1
    
    return config


def create_dataset_with_n_groups(config: EntityBindingTaskConfig, num_groups: int, size: int) -> CounterfactualDataset:
    """
    Create a dataset with exactly num_groups active groups.
    
    Both the input and counterfactual input are randomly generated independently
    to maximize diversity in the training set.
    
    Args:
        config: Task configuration
        num_groups: Number of active groups to use
        size: Number of examples to generate
        
    Returns:
        CounterfactualDataset with the specified number of groups
    """
    from causalab.tasks.entity_binding.causal_models import create_direct_causal_model
    
    config.fixed_active_groups = num_groups
    def generator():
        # Sample input and ensure it has exactly num_groups
        max_attempts = 100
        for attempt in range(max_attempts):
            input_sample = sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)
        
        # Regenerate raw_input with correct active_groups
        model = create_direct_causal_model(config)
        model.new_raw_input(input_sample)
        
        # Preserve query_indices and answer_index from original input
        # This ensures we ask about the same position within the group (e.g., first entity, second entity)
        original_query_indices = input_sample.get("query_indices", (0,))
        # Ensure query_indices is a tuple (not a list) for consistency
        if isinstance(original_query_indices, list):
            original_query_indices = tuple(original_query_indices)
        original_answer_index = input_sample.get("answer_index", 0)
        
        # Randomly generate counterfactual input independently (not by swapping)
        # This ensures maximum diversity in the training set
        for attempt in range(max_attempts):
            counterfactual_input = sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)
            if counterfactual_input["active_groups"] == num_groups:
                break
        else:
            # Force active_groups if we couldn't get it naturally
            counterfactual_input["active_groups"] = num_groups
            # Ensure query_group is valid
            query_group_cf = counterfactual_input.get("query_group", 0)
            if query_group_cf >= num_groups:
                counterfactual_input["query_group"] = query_group_cf % num_groups
        
        # IMPORTANT: Preserve query_indices and answer_index from original input
        # This ensures we ask about the same position within the group (e.g., first entity, second entity)
        counterfactual_input["answer_index"] = original_answer_index
        
        # Regenerate raw_input for counterfactual with correct active_groups
        # Ensure query_indices is explicitly set as a tuple before calling new_raw_input
        counterfactual_input["query_indices"] = original_query_indices  # Already a tuple from above
        model.new_raw_input(counterfactual_input)
        
        return {"input": input_sample, "counterfactual_inputs": [counterfactual_input]}
    
    return CounterfactualDataset.from_sampler(size, generator, id=f"entity_binding_{num_groups}groups")


def checker(neural_output, causal_output):
    """Check if neural network output matches causal model output."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


def filter_dataset_with_accuracy_check(
    dataset: CounterfactualDataset,
    pipeline: LMPipeline,
    causal_model,
    min_accuracy: float = 0.8,
    batch_size: int = 32,
    verbose: bool = True
):
    """
    Filter dataset and check if accuracy meets threshold.
    
    Args:
        dataset: Dataset to filter
        pipeline: Language model pipeline
        causal_model: Causal model for evaluation
        min_accuracy: Minimum accuracy threshold (default 0.8)
        batch_size: Batch size for filtering
        verbose: Whether to print progress
        
    Returns:
        Tuple of (filtered_dataset, stats_dict)
        stats_dict contains: original_size, filtered_size, accuracy, passed_threshold
        
    Raises:
        ValueError: If accuracy is below threshold and we should suggest a larger model
    """
    if verbose:
        print(f"Filtering dataset ({len(dataset)} examples)...")
    
    filtered_dataset = filter_dataset(
        dataset=dataset,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=batch_size,
        validate_counterfactuals=True
    )
    accuracy = len(filtered_dataset) / len(dataset) if len(dataset) > 0 else 0.0
    
    stats = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "accuracy": accuracy,
        "passed_threshold": accuracy >= min_accuracy
    }
    
    if verbose:
        print(f"  Original size: {stats['original_size']}")
        print(f"  Filtered size: {stats['filtered_size']}")
        print(f"  Accuracy: {accuracy:.1%}")
    
    if accuracy < min_accuracy:
        error_msg = (
            f"Task accuracy ({accuracy:.1%}) is below threshold ({min_accuracy:.1%}). "
            f"Only {stats['filtered_size']}/{stats['original_size']} examples passed filtering. "
            "Please use a larger model."
        )
        if verbose:
            print(f"\n⚠ WARNING: {error_msg}\n")
        raise ValueError(error_msg)
    
    return filtered_dataset, stats


def run_vanilla_interchange(
    causal_model,
    interchange_target: Dict[int, Any],  # Dict[layer, InterchangeTarget]
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    batch_size: int = 32,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run vanilla interchange interventions (no training, direct activation swapping).
    
    This function runs vanilla interchange interventions on each layer by directly
    swapping activations between base and counterfactual inputs, without any
    learned transformation.
    
    Args:
        causal_model: Causal model for generating expected outputs
        interchange_target: Dict mapping layer number to InterchangeTarget
        train_dataset_path: Path to training dataset directory
        test_dataset_path: Path to test dataset directory
        pipeline: LMPipeline object
        target_variable_group: Tuple of target variable names
        output_dir: Output directory for results
        metric: Function to compare neural output with expected output
        batch_size: Batch size for evaluation
        save_results: Whether to save results to disk
        verbose: Whether to print progress
        
    Returns:
        Dictionary with structure:
            - train_scores: Dict[int, float] (layer -> score)
            - test_scores: Dict[int, float] (layer -> score)
            - metadata: experiment configuration and summary
            - output_paths: paths to saved files
    """
    from pathlib import Path
    
    # Load datasets
    train_hf_dataset = load_from_disk(train_dataset_path)
    if not isinstance(train_hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(train_hf_dataset).__name__}")
    train_dataset = CounterfactualDataset(
        dataset=train_hf_dataset, id="train"
    )
    
    test_hf_dataset = load_from_disk(test_dataset_path)
    if not isinstance(test_hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(test_hf_dataset).__name__}")
    test_dataset = CounterfactualDataset(
        dataset=test_hf_dataset, id="test"
    )
    
    train_scores = {}
    test_scores = {}
    
    # Run interventions for each layer
    layers = sorted(interchange_target.keys())
    pbar = tqdm(layers, desc="Running vanilla interchange", disable=not verbose)
    
    for layer in pbar:
        target = interchange_target[layer]
        key = (layer,)
        
        if verbose:
            pbar.set_description(f"Layer {layer}: train")
        
        # Run interventions on train data
        train_raw_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=train_dataset,
                interchange_target=target,
                batch_size=batch_size,
                output_scores=False,
            )
        }
        
        # Score train results
        train_eval = causal_score_intervention_outputs(
            raw_results=train_raw_results,
            dataset=train_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )
        
        if verbose:
            pbar.set_description(f"Layer {layer}: test")
        
        # Run interventions on test data
        test_raw_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=test_dataset,
                interchange_target=target,
                batch_size=batch_size,
                output_scores=False,
            )
        }
        
        # Score test results
        test_eval = causal_score_intervention_outputs(
            raw_results=test_raw_results,
            dataset=test_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )
        
        train_scores[layer] = train_eval["results_by_key"][key]["avg_score"]
        test_scores[layer] = test_eval["results_by_key"][key]["avg_score"]
    
    pbar.close()
    
    # Find best layer
    best_layer = max(test_scores, key=lambda k: test_scores[k])
    avg_test_score = float(np.mean(list(test_scores.values())))
    
    # Create metadata
    model_name = getattr(pipeline, "model_or_name", None)
    metadata = {
        "experiment_type": "vanilla_interchange",
        "model": model_name,
        "train_dataset_path": train_dataset_path,
        "test_dataset_path": test_dataset_path,
        "target_variable_group": list(target_variable_group),
        "num_layers": len(interchange_target),
        "layers": layers,
        "best_layer": best_layer,
        "best_test_score": test_scores[best_layer],
        "avg_test_score": avg_test_score,
        "batch_size": batch_size,
    }
    
    # Save results if requested
    output_paths = {}
    if save_results:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_dir_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        output_paths["metadata"] = str(metadata_path)
        
        # Save scores
        scores = {
            "train_scores": {str(k): v for k, v in train_scores.items()},
            "test_scores": {str(k): v for k, v in test_scores.items()},
        }
        scores_path = output_dir_path / "scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        output_paths["scores"] = str(scores_path)
    
    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "metadata": metadata,
        "output_paths": output_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hypothesis testing for entity binding with vanilla interchange interventions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID (default: gpt2)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (default: use cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Dataset size to generate (default: 1024)",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="Number of entity groups (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for filtering and evaluation (default: 32)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.8,
        help="Minimum accuracy threshold (default: 0.8)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: reduced dataset size and batch size",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer numbers to test (e.g., '0,5,10' or '0-5'). If not specified, all layers are tested.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (sets HF_HOME env var, must be set before imports)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="action",
        choices=["action", "filling_liquids"],
        help="Task type: 'action' (person put object in location) or 'filling_liquids' (person fills container with liquid) (default: action)",
    )
    
    args = parser.parse_args()
    
    # Set HF_HOME if provided (fallback for when argparse processes it)
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        print(f"Using HuggingFace cache directory: {args.hf_cache_dir}")
    
    # Test mode overrides
    if args.test:
        args.size = 16
        args.batch_size = 8
        print("\n*** TEST MODE: size=16, batch_size=8 ***\n")
    
    # Parse layers argument
    selected_layers = None
    if args.layers:
        try:
            selected_layers = []
            for part in args.layers.split(','):
                part = part.strip()
                if '-' in part:
                    # Range like "0-5"
                    start, end = map(int, part.split('-'))
                    selected_layers.extend(range(start, end + 1))
                else:
                    # Single layer number
                    selected_layers.append(int(part))
            selected_layers = sorted(set(selected_layers))  # Remove duplicates and sort
        except ValueError as e:
            print(f"Error parsing --layers argument: {e}")
            print("Expected format: comma-separated list (e.g., '0,5,10') or range (e.g., '0-5')")
            return 1
    
    # Auto-generate output path
    if args.output is None:
        test_suffix = "_test" if args.test else ""
        args.output = f"outputs/{args.model.replace('/', '_')}{test_suffix}"
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    print("=" * 70)
    print("Entity Binding Hypothesis Testing with Vanilla Interchange")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Dataset size: {args.size}")
    print(f"  Number of groups: {args.num_groups}")
    print(f"  Entities per group: 3")
    print(f"  Task type: {args.task_type}")
    print(f"  Output directory: {output_dir}")
    print(f"  Test mode: {args.test}")
    if selected_layers is not None:
        print(f"  Selected layers: {selected_layers}")
    print()
    
    # Step 1: Create task configuration
    print("Step 1: Creating task configuration...")
    config = create_custom_config(task_type=args.task_type)
    print(f"  Task type: {args.task_type}")
    print(f"  Max groups: {config.max_groups}")
    print(f"  Entities per group: {config.max_entities_per_group}")
    print(f"  Template: {config.statement_template}")
    print()
    
    # Step 2: Generate dataset
    print(f"Step 2: Generating dataset with {args.num_groups} groups...")
    dataset = create_dataset_with_n_groups(config, args.num_groups, args.size)
    print(f"  Generated {len(dataset)} examples")
    
    # Save partial result: raw dataset
    raw_dataset_path = output_dir / "raw_dataset"
    dataset.dataset.save_to_disk(str(raw_dataset_path))
    print(f"  Saved raw dataset to {raw_dataset_path}")
    
    # Save partial summary
    partial_summary = {
        "model": args.model,
        "device": device,
        "num_groups": args.num_groups,
        "entities_per_group": 3,
        "task_type": args.task_type,
        "dataset_size": args.size,
        "raw_dataset_size": len(dataset),
        "step": "dataset_generated"
    }
    partial_summary_path = output_dir / "partial_summary.json"
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Show example
    # Create causal model to compute raw_output
    causal_model_for_display = create_positional_causal_model(config)
    if len(dataset) > 0:
        print("Example from dataset:")
        for i, example in zip(range(5), dataset):
            # Compute raw_output for input if not present
            input_dict = example['input'].copy()
            if 'raw_output' not in input_dict:
                result = causal_model_for_display.run_forward(input_dict)
                input_dict['raw_output'] = result.get('raw_output', 'N/A')
            
            # Compute raw_output for counterfactual if not present
            counterfactual_dict = example['counterfactual_inputs'][0].copy()
            if 'raw_output' not in counterfactual_dict:
                result = causal_model_for_display.run_forward(counterfactual_dict)
                counterfactual_dict['raw_output'] = result.get('raw_output', 'N/A')
            
            print(f"  Input:  {input_dict['raw_input']}...")
            print(f"  Expected Answer: {input_dict['raw_output']}")
            print(f"  Counter: {counterfactual_dict['raw_input']}...")
            print(f"  Expected Answer: {counterfactual_dict['raw_output']}")
            print()
  
    # Step 3: Load model and causal model
    print(f"Step 3: Loading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=2,
        device=device,
        max_length=None,  # Use dynamic padding (pad to longest in batch) instead of fixed 512
    )
    pipeline.tokenizer.padding_side = "left"
    num_layers = pipeline.get_num_layers()
    print(f"  Model loaded ({num_layers} layers)")
    
    # Layer info will be printed after validation in Step 6
    print()
    
    causal_model = create_positional_causal_model(config)
    print(f"  Causal model: {causal_model.id}")
    print()
    
    # Step 4: Filter dataset
    print("Step 4: Filtering dataset...")
    try:
        filtered_dataset, filter_stats = filter_dataset_with_accuracy_check(
            dataset,
            pipeline,
            causal_model,
            min_accuracy=args.min_accuracy,
            batch_size=args.batch_size,
            verbose=True
        )
        
        # Save partial result: filtered dataset
        filtered_dataset_path = output_dir / "filtered_dataset"
        filtered_dataset.dataset.save_to_disk(str(filtered_dataset_path))
        print(f"  Saved filtered dataset to {filtered_dataset_path}")
        
        # Update partial summary
        partial_summary.update({
            "filter_stats": filter_stats,
            "filtered_dataset_size": len(filtered_dataset),
            "step": "dataset_filtered"
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        # Save error state to partial summary
        partial_summary.update({
            "error": str(e),
            "step": "filtering_failed"
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        return 1
    
    # Step 5: Split into train/test (80/20)
    print("Step 5: Splitting dataset into train/test...")
    train_size = int(len(filtered_dataset) * 0.8)
    test_size = len(filtered_dataset) - train_size
    
    train_dataset = CounterfactualDataset(
        dataset=filtered_dataset.dataset.select(range(train_size)),
        id="train"
    )
    test_dataset = CounterfactualDataset(
        dataset=filtered_dataset.dataset.select(range(train_size, len(filtered_dataset))),
        id="test"
    )
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Test size: {len(test_dataset)}")
    print()
    
    # Save datasets
    train_path = output_dir / "train_dataset"
    test_path = output_dir / "test_dataset"
    train_dataset.dataset.save_to_disk(str(train_path))
    test_dataset.dataset.save_to_disk(str(test_path))
    print(f"  Saved datasets to {output_dir}")
    
    # Update partial summary
    partial_summary.update({
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "step": "train_test_split"
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Step 6: Setup for vanilla interchange
    print("Step 6: Setting up vanilla interchange...")
    
    # Create token position (last token)
    def last_token_indexer(input_dict, is_original=True):
        return get_last_token_index(input_dict["raw_input"], pipeline)
    
    token_position = TokenPosition(
        last_token_indexer,
        pipeline,
        id="last_token"
    )
    token_positions = [token_position]
    
    # Build residual stream targets for all layers (or selected layers)
    if selected_layers is not None:
        # Validate selected layers are within valid range
        invalid_layers = [l for l in selected_layers if l < 0 or l >= num_layers]
        if invalid_layers:
            print(f"Error: Invalid layer numbers: {invalid_layers}")
            print(f"Valid layer range: 0-{num_layers-1}")
            return 1
        layers = selected_layers
        print(f"  Testing selected layers: {layers}")
    else:
        layers = list(range(num_layers))
        print(f"  Testing all layers: {layers}")
    residual_targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_layer",
        target_output=False,  # Use block_input (resid_pre) to match original paper
    )
    
    # Convert to {layer: target} format
    residual_targets_by_layer = {key[0]: target for key, target in residual_targets.items()}
    
    print(f"  Created targets for {len(residual_targets_by_layer)} layers")
    print(f"  Token position: last_token")
    print(f"  Target variable: positional_query_group")
    
    # Save partial result: Intervention configuration
    intervention_config = {
        "num_layers": num_layers,
        "layer_start": layers[0] if layers else 0,
        "layer_end": layers[-1] + 1 if layers else num_layers,
        "layers": layers,
        "token_position": "last_token",
        "target_variable": "positional_query_group",
        "intervention_type": "vanilla",
    }
    intervention_config_path = output_dir / "intervention_config.json"
    with open(intervention_config_path, "w") as f:
        json.dump(intervention_config, f, indent=2)
    print(f"  Saved intervention configuration to {intervention_config_path}")
    
    # Update partial summary
    partial_summary.update({
        "num_layers": num_layers,
        "layer_start": layers[0] if layers else 0,
        "layer_end": layers[-1] + 1 if layers else num_layers,
        "intervention_type": "vanilla",
        "step": "intervention_setup_complete"
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Step 7: Run vanilla interchange
    print("Step 7: Running vanilla interchange intervention...")
    print()
    
    intervention_output_dir = output_dir / "vanilla_interchange"
    intervention_output_dir.mkdir(exist_ok=True)
    
    # Update partial summary before intervention
    partial_summary.update({
        "step": "vanilla_intervention_started",
        "intervention_type": "vanilla",
        "intervention_output_dir": str(intervention_output_dir)
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    
    try:
        result = run_vanilla_interchange(
            causal_model=causal_model,
            interchange_target=residual_targets_by_layer,
            train_dataset_path=str(train_path),
            test_dataset_path=str(test_path),
            pipeline=pipeline,
            target_variable_group=("positional_query_group",),
            output_dir=str(intervention_output_dir),
            metric=checker,
            batch_size=args.batch_size,
            save_results=True,
            verbose=True,
        )
        
        print()
        print("✓ Vanilla interchange intervention complete!")
        print()
        
        # Add num_groups and task_type to result metadata
        result["metadata"]["num_groups"] = args.num_groups
        result["metadata"]["task_type"] = args.task_type
        
        # Update partial summary with IIA scores for each layer
        partial_summary.update({
            "step": "vanilla_intervention_complete",
            "best_layer": result["metadata"]["best_layer"],
            "best_test_score": result["metadata"]["best_test_score"],
            "avg_test_score": result["metadata"]["avg_test_score"],
            "test_scores_by_layer": result["test_scores"],  # IIA accuracy for each layer
            "train_scores_by_layer": result["train_scores"],
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print(f"  Updated partial_summary.json with IIA scores for all layers")
        print()
        
        # Step 8: Save results summary
        print("Step 8: Saving results summary...")
        
        summary = {
            "model": args.model,
            "device": device,
            "num_groups": args.num_groups,
            "entities_per_group": 3,
            "task_type": args.task_type,
            "dataset_size": args.size,
            "filter_stats": filter_stats,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "num_layers": num_layers,
            "layer_start": layers[0] if layers else 0,
            "layer_end": layers[-1] + 1 if layers else num_layers,
            "layers": layers,
            "intervention_type": "vanilla",
            "target_variable": "positional_query_group",
            "best_layer": result["metadata"]["best_layer"],
            "best_test_score": result["metadata"]["best_test_score"],
            "avg_test_score": result["metadata"]["avg_test_score"],
            "test_scores_by_layer": result["test_scores"],
            "train_scores_by_layer": result["train_scores"],
        }
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save full result
        result_path = output_dir / "vanilla_interchange_result.pkl"
        with open(result_path, "wb") as f:
            pickle.dump(result, f)
        
        print(f"  Summary saved to: {summary_path}")
        print(f"  Full result saved to: {result_path}")
        print()
        
        # Print summary
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Intervention type: vanilla")
        print(f"Filter accuracy: {filter_stats['accuracy']:.1%}")
        print(f"Best layer: {result['metadata']['best_layer']}")
        print(f"Best test score (IIA accuracy): {result['metadata']['best_test_score']:.3f}")
        print(f"Average test score: {result['metadata']['avg_test_score']:.3f}")
        print()
        print(f"Test scores by layer:")
        for layer in sorted(result["test_scores"].keys()):
            score = result["test_scores"][layer]
            marker = " ★" if layer == result["metadata"]["best_layer"] else ""
            print(f"  Layer {layer:2d}: {score:.3f}{marker}")
        print()
        print(f"All results saved to: {output_dir}")
        print()
        print("Saved files:")
        print(f"  - Summary: {summary_path}")
        print(f"  - Full result: {result_path}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during vanilla interchange intervention: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error state to partial summary
        partial_summary.update({
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "step": "vanilla_intervention_failed",
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print(f"\n  Partial results saved to: {output_dir}")
        print(f"  Check partial_summary.json for progress details")
        return 1


if __name__ == "__main__":
    sys.exit(main())

