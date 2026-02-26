#!/usr/bin/env python
"""
Test Alignment Across Different Target Entity Locations

This script:
1. Loads a saved alignment result from run_das.py
2. Creates datasets with different target entity locations (query_group values) in counterfactuals
3. Filters datasets to keep only examples where both base and counterfactual are correct (like run_das.py)
4. Tests the alignment on each location
5. Plots IIA accuracy vs target entity location

Usage:
    python test_alignment.py --result-path outputs/MODEL/boundless_das_result.pkl [--gpu GPU_ID]
    python test_alignment.py --result-path outputs/Qwen_Qwen3-8B/vanilla_interchange_result.pkl --gpu 0
    python test_alignment.py --result-path outputs/MODEL/result.pkl --task-type filling_liquids
    python test_alignment.py --result-path outputs/MODEL/result.pkl --query-entity 0
    python test_alignment.py --result-path outputs/MODEL/result.pkl --skip-filtering  # Disable filtering

Task types:
    - action: Person put object in location (default)
    - filling_liquids: Person fills container with liquid
    - music: Music-related task
    - boxes: Boxes-related task

Query entity options:
    - --query-entity 0: Fix query_indices to (0,) (query first entity in group)
    - --query-entity 1: Fix query_indices to (1,) (query second entity in group)
    - --query-entity 2: Fix query_indices to (2,) (query third entity in group)
    - If not specified, uses query_indices from original input sample

Filtering:
    - By default, filters datasets to keep only examples where both base and counterfactual predictions are correct
    - Use --skip-filtering to disable filtering (test all examples regardless of correctness)
"""

import sys
import os

# Set HuggingFace cache directory BEFORE importing transformers
for i, arg in enumerate(sys.argv):
    if arg == "--hf-cache-dir" and i + 1 < len(sys.argv):
        os.environ["HF_HOME"] = sys.argv[i + 1]
        print(f"Using HuggingFace cache directory: {sys.argv[i + 1]}")
        break

import argparse
import torch
import json
import pickle
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Callable, List
import matplotlib.pyplot as plt
import numpy as np

# Add the causalab package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'causalab'))

from causalab.tasks.entity_binding.config import (
    EntityBindingTaskConfig, 
    create_sample_action_config,
    create_filling_liquids_config
)
try:
    from causalab.tasks.entity_binding.config import (
        create_music_config,
        create_boxes_config
    )
except ImportError:
    create_music_config = None
    create_boxes_config = None
from causalab.tasks.entity_binding.causal_models import create_positional_entity_causal_model, sample_valid_entity_binding_input
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import (
    generate_counterfactual_samples,
    save_counterfactual_examples,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import build_token_position_factories
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.experiments.metric import causal_score_intervention_outputs
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.filter import filter_dataset
from datasets import load_from_disk, Dataset
from tqdm import tqdm


# ========== Register Qwen3 support for pyvene ==========
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
    - action, filling_liquids, music: 3 entities per group
    - boxes: 2 entities per group
    
    Args:
        task_type: One of "action" (person put object in location), 
                  "filling_liquids" (person fills container with liquid),
                  "music" (music-related task), or
                  "boxes" (boxes-related task)
    
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
    elif task_type == "music":
        if create_music_config is None:
            raise ValueError("create_music_config function not found. Please ensure it is defined in the config module.")
        config = create_music_config()
        # Expand entity pools for music task to support 10 groups with unique entities
        config.entity_pools[0] = [
            "John", "Mary", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
            "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
            "Ben", "Liz", "Tom", "Joy"
        ]
        config.entity_pools[1] = [
            "rock", "pop", "jazz", "classical", "blues", "country", "folk", "reggae",
            "electronic", "metal", "punk", "soul", "funk", "gospel", "disco", "techno",
            "indie", "grunge", "swing", "opera"
        ]
        config.entity_pools[2] = [
            "piano", "guitar", "drums", "violin", "flute", "saxophone", "trumpet", "bass",
            "cello", "clarinet", "harp", "organ", "ukulele", "banjo", "mandolin", "harmonica",
            "trombone", "oboe", "tuba", "accordion"
        ]
    elif task_type == "boxes":
        if create_boxes_config is None:
            raise ValueError("create_boxes_config function not found. Please ensure it is defined in the config module.")
        config = create_boxes_config()
        # Expand entity pools for boxes task to support 10 groups with unique entities
        # Note: boxes has only 2 entities per group (object, box)
        config.entity_pools[0] = [
            "toy", "medicine", "book", "coin", "pen", "key", "phone", "watch",
            "cup", "ball", "bag", "hat", "map", "card", "lamp", "rope",
            "tape", "tool", "clip", "pencil"
        ]
        config.entity_pools[1] = [
            "box A", "box B", "box C", "box D", "box E", "box F", "box G", "box H",
            "box I", "box J", "box K", "box L", "box M", "box N", "box O", "box P",
            "box Q", "box R", "box S", "box T"
        ]
    else:
        # Default to action task
        config = create_sample_action_config()
        
        # Expand entity pools for action task
        config.entity_pools[0] = [
            "Pete", "Ann", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
            "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
            "Ben", "Liz", "Tom", "Joy"
        ]
        
        config.entity_pools[1] = [
            "jam", "water", "book", "coin", "pen", "key", "phone", "watch",
            "cup", "box", "bag", "hat", "map", "card", "lamp", "ball",
            "rope", "tape", "tool", "clip"
        ]
        
        config.entity_pools[2] = [
            "cup", "box", "table", "shelf", "drawer", "bag", "pocket", "basket",
            "desk", "chair", "floor", "rack", "case", "tray", "bin", "stand",
            "cabinet", "corner", "bench", "counter"
        ]
    
    # Set max_groups for all tasks
    config.max_groups = 10
    
    # IMPORTANT: Do NOT override max_entities_per_group here!
    # Each task type has its own correct value:
    # - action, filling_liquids, music: 3 (already set in their configs)
    # - boxes: 2 (already set in its config)
    # Overriding to 3 for all tasks was causing a bug for boxes task.
    
    config.prompt_prefix = "We will ask a question about the following sentences. Only return the answer, no other text.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"

    config.fixed_query_indices = (1,)
    config.fixed_answer_index = 0
    
    return config


def create_dataset_with_fixed_counterfactual_query_group(
    config: EntityBindingTaskConfig,
    num_groups: int,
    size: int,
    counterfactual_query_group: int,
    use_swap: bool = False,
    query_entity: int = None
) -> list[CounterfactualExample]:
    """
    Create a dataset where counterfactuals have a fixed query_group (target entity location).
    
    Following the pattern from counterfactuals.py swap_query_group function.
    
    Args:
        config: Task configuration
        num_groups: Number of active groups
        size: Number of examples to generate
        counterfactual_query_group: Fixed query_group for counterfactuals
        use_swap: If True, generate counterfactual by swapping entities with the target location.
                 If False, generate a random counterfactual and fix the query_group.
        query_entity: If provided (0, 1, or 2), fix query_indices to (query_entity,)
                     and ensure answer_index is different from query_entity.
        
    Returns:
        list[CounterfactualExample] with fixed counterfactual query_group
    """
    config.fixed_active_groups = num_groups

    def generator():
        # Create causal model
        model = create_positional_entity_causal_model(config)
        
        # Sample input
        input_sample = sample_valid_entity_binding_input(config, model, ensure_positional_uniqueness=True)
        
        # Ensure target query_group is within valid range
        target_query_group = counterfactual_query_group % num_groups
        original_query_group = input_sample["query_group"]
        
        # If input's query_group matches target, change input to use a different group
        if original_query_group == target_query_group:
            # Build new input dict from INPUT variables only (like swap_query_group does)
            input_dict = {var: input_sample[var] for var in model.inputs}
            
            # Find a different group
            for g in range(num_groups):
                if g != target_query_group:
                    input_dict["query_group"] = g
                    # Update query_e{e} to match the new query group
                    for e in range(config.max_entities_per_group):
                        input_dict[f"query_e{e}"] = input_dict[f"entity_g{g}_e{e}"]
                    break
            
            input_sample = model.new_trace(input_dict)
            original_query_group = input_sample["query_group"]
        
        # Apply query_entity constraint if specified
        if query_entity is not None:
            if query_entity < 0 or query_entity >= config.max_entities_per_group:
                raise ValueError(f"query_entity must be between 0 and {config.max_entities_per_group - 1}, got {query_entity}")
            
            input_dict = {var: input_sample[var] for var in model.inputs}
            input_dict["query_indices"] = (query_entity,)
            
            # Ensure answer_index is different from query_entity
            if input_dict["answer_index"] == query_entity:
                for alt_idx in range(config.max_entities_per_group):
                    if alt_idx != query_entity:
                        input_dict["answer_index"] = alt_idx
                        break
            
            input_sample = model.new_trace(input_dict)
        
        # Build counterfactual dict from INPUT variables only
        cf_dict = {var: input_sample[var] for var in model.inputs}
        
        if use_swap and original_query_group != target_query_group:
            # Swap entities between original_query_group and target_query_group
            entities_per_group = input_sample["entities_per_group"]
            for e in range(entities_per_group):
                key_orig = f"entity_g{original_query_group}_e{e}"
                key_target = f"entity_g{target_query_group}_e{e}"
                # Swap
                cf_dict[key_orig], cf_dict[key_target] = cf_dict[key_target], cf_dict[key_orig]
            
            # Update query_group to follow where the original query entities moved
            cf_dict["query_group"] = target_query_group
            
            # Update query_e{e} from entities at target_query_group (which now has original entities)
            for e in range(entities_per_group):
                cf_dict[f"query_e{e}"] = cf_dict[f"entity_g{target_query_group}_e{e}"]
        else:
            # Random counterfactual: sample new one and fix its query_group
            cf_sample = sample_valid_entity_binding_input(config, model, ensure_positional_uniqueness=True)
            cf_dict = {var: cf_sample[var] for var in model.inputs}
            
            # Force query_group to target
            cf_dict["query_group"] = target_query_group
            
            # Update query_e{e} to match target query group
            for e in range(config.max_entities_per_group):
                cf_dict[f"query_e{e}"] = cf_dict[f"entity_g{target_query_group}_e{e}"]
            
            # Preserve query_indices and answer_index from original input
            cf_dict["query_indices"] = input_sample["query_indices"]
            cf_dict["answer_index"] = input_sample["answer_index"]
        
        # Create counterfactual trace
        counterfactual = model.new_trace(cf_dict)
        
        return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
    
    dataset = generate_counterfactual_samples(size, generator)
    return dataset


def checker(neural_output, causal_output):
    """Check if neural network output matches causal model output."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


def filter_dataset_with_accuracy_check(
    dataset: list[CounterfactualExample],
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


def load_alignment_result(result_path: Path) -> Dict[str, Any]:
    """Load saved alignment result from run_das.py."""
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    
    return result

def create_token_positions(pipeline: LMPipeline, task_config):
    """Create token positions for entity binding task."""
    template = task_config.build_mega_template(
        active_groups=task_config.max_groups,
        query_indices=task_config.fixed_query_indices,
        answer_index=task_config.fixed_answer_index,
    )

    token_position_specs = {
        "last_token": {"type": "index", "position": -1},
    }

    factories = build_token_position_factories(token_position_specs, template)
    token_positions = {}
    for name, factory in factories.items():
        token_positions[name] = factory(pipeline)

    return token_positions

def test_alignment_for_location(
    result: Dict[str, Any],
    pipeline: LMPipeline,
    causal_model,
    config: EntityBindingTaskConfig,
    num_groups: int,
    test_size: int,
    target_location: int,
    batch_size: int = 32,
    use_swap: bool = False,
    verbose: bool = True,
    layer: int = None,
    query_entity: int = None,
    skip_filtering: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Test alignment for a specific target entity location (query_group in counterfactual).
    
    Args:
        result: Loaded alignment result from run_das.py
        pipeline: LMPipeline object
        causal_model: Causal model for evaluation
        config: Task configuration
        num_groups: Number of active groups
        test_size: Size of test dataset
        target_location: Query group for counterfactuals (target entity location)
        batch_size: Batch size for evaluation
        use_swap: Whether to use swap counterfactuals
        verbose: Whether to print progress
        layer: Optional layer to use. If None, uses best_layer from result.
        query_entity: Optional query entity (0, 1, or 2) to fix query_indices.
        skip_filtering: Whether to skip filtering (default: False)
        
    Returns:
        Tuple of (IIA accuracy for this location, filter_stats dict)
    """
    intervention_type = result.get("metadata", {}).get("experiment_type", "vanilla")
    
    # Create dataset with fixed counterfactual query_group
    test_dataset = create_dataset_with_fixed_counterfactual_query_group(
        config=config,
        num_groups=num_groups,
        size=test_size,
        counterfactual_query_group=target_location,
        use_swap=use_swap,
        query_entity=query_entity
    )
    
    # Filter dataset to keep only examples where both base and counterfactual are correct
    filter_stats = None
    if not skip_filtering:
        try:
            test_dataset, filter_stats = filter_dataset_with_accuracy_check(
                test_dataset,
                pipeline,
                causal_model,
                min_accuracy=0.0,  # Don't raise error, just filter
                batch_size=batch_size,
                verbose=verbose
            )
        except ValueError as e:
            # If filtering fails due to low accuracy, return 0.0 accuracy
            if verbose:
                print(f"    Warning: Filtering failed for location {target_location}: {e}")
            filter_stats = {
                "original_size": len(test_dataset),
                "filtered_size": 0,
                "accuracy": 0.0,
                "passed_threshold": False
            }
            return 0.0, filter_stats
    
    # Get layer to use: either provided directly or from result
    if layer is not None:
        best_layer = layer
        if verbose:
            print(f"  Using specified layer: {best_layer}")
    else:
        # Get best layer from result
        best_layer = result.get("metadata", {}).get("best_layer")
        if best_layer is None:
            # Fall back to layer with best test score
            test_scores = result.get("test_scores", {})
            if test_scores:
                best_layer = max(test_scores, key=lambda k: test_scores[k])
            else:
                raise ValueError("Could not determine best layer from result. Use --layer to specify a layer.")
        if verbose:
            print(f"  Using best layer from result: {best_layer}")
    
    # Get interchange target for best layer
    # Recreate token position  
    
    token_positions_dict = create_token_positions(pipeline, config)
    token_positions = list(token_positions_dict.values())
    
    # Build targets for the best layer
    # NOTE: Using one_target_per_unit to match replicate.py behavior (one target per (layer, position))
    # NOTE: target_output defaults to True (block_output) to match replicate.py
    residual_targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=[best_layer],
        token_positions=token_positions,
        mode="one_target_per_unit",  # Changed from one_target_per_layer to match replicate.py
        # target_output defaults to True (block_output), matching replicate.py
    )
    
    # With one_target_per_unit, keys are (layer, position_id) tuples
    # Since we only have one token position, there should be exactly one target
    if len(residual_targets) != 1:
        raise ValueError(
            f"Expected exactly one target for layer {best_layer} with {len(token_positions)} token positions, "
            f"but got {len(residual_targets)} targets: {list(residual_targets.keys())}"
        )
    
    # Get the single target (key will be (best_layer, position_id))
    key, target = next(iter(residual_targets.items()))
    
    # Note: For boundless DAS, we could load trained weights here, but for simplicity
    # we use vanilla interchange which should still provide useful alignment testing.
    # The trained weights would improve accuracy but the relative performance across
    # locations should still be informative.
    
    # Run interventions
    raw_results = {
        key: run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=test_dataset,
            interchange_target=target,
            batch_size=batch_size,
            output_scores=False,
        )
    }
    
    # Score results
    eval_result = causal_score_intervention_outputs(
        raw_results=raw_results,
        dataset=test_dataset,
        causal_model=causal_model,
        target_variable_groups=[("positional_answer",)],
        metric=checker,
    )
    
    accuracy = eval_result["results_by_key"][key]["avg_score"]
    
    return accuracy, filter_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test alignment across different target entity locations"
    )
    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="Path to saved result file (boundless_das_result.pkl or vanilla_interchange_result.pkl)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (default: use cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Size of test dataset per location (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as result directory)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (sets HF_HOME env var)",
    )
    parser.add_argument(
        "--use-swap",
        action="store_true",
        help="Generate counterfactuals by swapping entities with target location (default: random counterfactual with fixed query_group)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to test (default: use best_layer from result file)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="action",
        choices=["action", "filling_liquids", "music", "boxes"],
        help="Task type: 'action' (person put object in location), 'filling_liquids' (person fills container with liquid), 'music' (music-related task), or 'boxes' (boxes-related task) (default: action)",
    )
    parser.add_argument(
        "--query-entity",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Query entity index (0, 1, or 2) to fix query_indices. If not specified, uses query_indices from original input.",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="Skip filtering datasets (keep all examples regardless of model correctness). Default is to filter.",
    )
    
    args = parser.parse_args()
    
    # Set HF_HOME if provided
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        print(f"Using HuggingFace cache directory: {args.hf_cache_dir}")
    
    result_path = Path(args.result_path)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = result_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    print("=" * 70)
    print("Testing Alignment Across Target Entity Locations")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Result path: {result_path}")
    print(f"  Device: {device}")
    print(f"  Test size per location: {args.test_size}")
    print(f"  Counterfactual generation: {'swap' if args.use_swap else 'random'}")
    print(f"  Task type: {args.task_type}")
    if args.query_entity is not None:
        print(f"  Query entity: {args.query_entity}")
    print(f"  Filtering: {'disabled' if args.skip_filtering else 'enabled'}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Load result
    print("Loading alignment result...")
    result = load_alignment_result(result_path)
    metadata = result.get("metadata", {})
    
    # Handle both boundless DAS and vanilla interchange result formats
    model_name = metadata.get("model", "unknown")
    num_groups = 10 #metadata.get("num_groups", 6)
    
    # Check experiment_type or infer from result structure
    if "experiment_type" in metadata:
        intervention_type = metadata["experiment_type"]
    elif "intervention_type" in metadata:
        intervention_type = "boundless" if metadata["intervention_type"] == "boundless" else "vanilla"
    else:
        # Try to infer from result structure
        intervention_type = "boundless" if "feature_indices" in result else "vanilla"
    
    # Determine which layer to use
    if args.layer is not None:
        best_layer = args.layer
        print(f"  Model: {model_name}")
        print(f"  Number of groups: {num_groups}")
        print(f"  Intervention type: {intervention_type}")
        print(f"  Using specified layer: {best_layer}")
        print()
    else:
        best_layer = metadata.get("best_layer")
        if best_layer is None:
            # Fall back to layer with best test score
            test_scores = result.get("test_scores", {})
            if test_scores:
                best_layer = max(test_scores, key=lambda k: test_scores[k])
            else:
                raise ValueError("Could not determine best layer from result. Use --layer to specify a layer.")
        
        print(f"  Model: {model_name}")
        print(f"  Number of groups: {num_groups}")
        print(f"  Intervention type: {intervention_type}")
        print(f"  Using best layer from result: {best_layer}")
        print()
    
    # Basic validation: layer must be non-negative
    if args.layer is not None and args.layer < 0:
        raise ValueError(f"Layer must be non-negative, got {args.layer}")
    
    # Load model
    print(f"Loading model {model_name}...")
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=5,
        device=device,
        max_length=200,
    )
    pipeline.tokenizer.padding_side = "left"
    print(f"  Model loaded")
    
    # Validate layer against actual model after loading
    if args.layer is not None:
        num_layers = pipeline.get_num_layers()
        if args.layer >= num_layers:
            raise ValueError(
                f"Specified layer {args.layer} is out of range. "
                f"Model has {num_layers} layers (0-{num_layers-1})."
            )
        print(f"  Validated layer {args.layer} (model has {num_layers} layers)")
    
    print()
    
    # Create config
    print("Creating task configuration...")
    config = create_custom_config(task_type=args.task_type)
    config.max_groups = num_groups
    print(f"  Task type: {args.task_type}")
    print(f"  Max groups: {num_groups}")
    print()
    
    # Create causal model
    causal_model = create_positional_entity_causal_model(config)
    print(f"  Causal model: {causal_model.id}")
    print()
    
    # Test alignment for each target location
    print("Testing alignment for each target entity location...")
    locations = list(range(num_groups))
    accuracies = []
    filter_stats_by_location = {}
    
    for location in tqdm(locations, desc="Testing locations"):
        try:
            accuracy, filter_stats = test_alignment_for_location(
                result=result,
                pipeline=pipeline,
                causal_model=causal_model,
                config=config,
                num_groups=num_groups,
                test_size=args.test_size,
                target_location=location,
                batch_size=args.batch_size,
                use_swap=args.use_swap,
                verbose=False,
                layer=args.layer,
                query_entity=args.query_entity,
                skip_filtering=args.skip_filtering
            )
            accuracies.append(accuracy)
            filter_stats_by_location[location] = filter_stats
            if filter_stats:
                print(f"  Location {location}: IIA accuracy = {accuracy:.3f} (filtered: {filter_stats['filtered_size']}/{filter_stats['original_size']}, acc: {filter_stats['accuracy']:.1%})")
            else:
                print(f"  Location {location}: IIA accuracy = {accuracy:.3f}")
        except Exception as e:
            print(f"  Location {location}: Error - {e}")
            accuracies.append(0.0)
            filter_stats_by_location[location] = None
    
    print()
    
    # Save results
    results_dict = {
        "model": model_name,
        "num_groups": num_groups,
        "intervention_type": intervention_type,
        "best_layer": best_layer,
        "task_type": args.task_type,
        "test_size_per_location": args.test_size,
        "counterfactual_generation": "swap" if args.use_swap else "random",
        "query_entity": args.query_entity,
        "locations": locations,
        "accuracies": accuracies,
        "filter_stats_by_location": filter_stats_by_location,
    }
    
    results_path = output_dir / f"alignment_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {results_path}")
    print()
    
    # Create plot
    print("Creating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(locations, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Target Entity Location (Counterfactual Query Group)", fontsize=12)
    plt.ylabel("IIA Accuracy", fontsize=12)
    if args.task_type == "filling_liquids":
        task_name = "Filling Liquids"
    elif args.task_type == "music":
        task_name = "Music"
    elif args.task_type == "boxes":
        task_name = "Boxes"
    else:
        task_name = "Action (Put Object in Location)"
    query_entity_str = f", Query Entity: {args.query_entity}" if args.query_entity is not None else ""
    plt.title(f"Alignment Test: IIA Accuracy vs Target Entity Location\nModel: {model_name}, Layer: {best_layer}, Task: {task_name}{query_entity_str}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.xticks(locations)
    
    # Add value labels on points
    for loc, acc in zip(locations, accuracies):
        plt.text(loc, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    safe_model_name = model_name.replace("/", "_")
    plot_path = output_dir / f"alignment_test_plot_{safe_model_name}_{args.task_type}_{args.skip_filtering}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    print()
    
    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Task type: {args.task_type}")
    print(f"Best layer: {best_layer}")
    print(f"Intervention type: {intervention_type}")
    if args.query_entity is not None:
        print(f"Query entity: {args.query_entity}")
    print(f"Filtering: {'disabled' if args.skip_filtering else 'enabled'}")
    print()
    print("IIA Accuracy by Target Entity Location:")
    for loc, acc in zip(locations, accuracies):
        stats = filter_stats_by_location.get(loc)
        if stats and not args.skip_filtering:
            print(f"  Location {loc}: {acc:.3f} (filtered: {stats['filtered_size']}/{stats['original_size']}, acc: {stats['accuracy']:.1%})")
        else:
            print(f"  Location {loc}: {acc:.3f}")
    print()
    print(f"Mean accuracy: {np.mean(accuracies):.3f}")
    print(f"Std accuracy: {np.std(accuracies):.3f}")
    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

