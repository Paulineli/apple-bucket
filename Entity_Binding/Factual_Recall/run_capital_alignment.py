#!/usr/bin/env python
"""
Capital Factual Recall Alignment

Find alignment of capital knowledge in an LLM using the prompt template:
"The capital of A is B." We ask the model to fill in B; A is a country/state/province.

This script:
1. Builds the capital causal model (get_capital_model)
2. Generates counterfactual data (base (A,B) vs counterfactual (A',B'))
3. Filters to keep only examples where both base and counterfactual are correct (like test_alignment.py)
4. Computes IIA at the token position of A across layers
5. Plots a heatmap of IIA by layer (like replicate.py)

Usage:
    python run_capital_alignment.py --model Qwen/Qwen3-8B [--gpu 0]
    python run_capital_alignment.py --model meta-llama/Llama-3.2-1B-Instruct --dataset-size 128 --batch-size 32
"""

import sys
import os

# Add causalab and Factual_Recall to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "causalab"))
sys.path.insert(0, _SCRIPT_DIR)

# Optional: set HuggingFace cache before importing transformers
for i, arg in enumerate(sys.argv):
    if arg == "--hf-cache-dir" and i + 1 < len(sys.argv):
        os.environ["HF_HOME"] = sys.argv[i + 1]
        break

import argparse
import random
from pathlib import Path

import torch

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.causal_utils import (
    generate_counterfactual_samples,
    save_counterfactual_examples,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import build_token_position_factories
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.interchange_score_grid import run_interchange_score_heatmap

from config import CapitalTaskConfig
from causal_models import create_capital_model


# ========== Register Qwen3 support for pyvene ==========
def register_qwen3_for_pyvene():
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import (
            type_to_module_mapping,
            type_to_dimension_mapping,
        )
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
        )

        if hasattr(qwen3_modeling, "Qwen3Model"):
            type_to_module_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_dimension_mapping
        if hasattr(qwen3_modeling, "Qwen3ForCausalLM"):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        print("Successfully registered Qwen3 support for pyvene")
    except ImportError:
        pass
    except Exception:
        pass


register_qwen3_for_pyvene()
# ========================================================


def checker(neural_output, causal_output):
    """Check if model output matches expected capital (causal output)."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


def capital_counterfactual_generator(config: CapitalTaskConfig, model):
    """Yield counterfactual examples: base (A,B) and counterfactual (A',B') with A' != A."""

    pairs = config.capital_pairs
    if len(pairs) < 2:
        raise ValueError("Need at least 2 (region, capital) pairs for counterfactuals")

    while True:
        (a, b) = random.choice(pairs)
        (a2, b2) = random.choice(pairs)
        if a == a2:
            continue
        base_trace = model.new_trace({"A": a, "B": b})
        cf_trace = model.new_trace({"A": a2, "B": b2})
        yield {
            "input": base_trace,
            "counterfactual_inputs": [cf_trace],
        }


def create_token_positions(
    pipeline: LMPipeline, config: CapitalTaskConfig, token_position: str
):
    """Create token positions for interventions.

    Supported modes:
    - token_position == \"A\": last token of variable A (region name).
      (A can be multi-token; we select a single token for pyvene compatibility.)
    - token_position == \"last_token\": last token of the full prompt (replicate.py-style).
    """
    template = config.prompt_template

    if token_position == "A":
        token_position_specs = {
            "position_A": {
                "type": "index",
                "position": -1,  # Last token of variable A
                "scope": {"variable": "A"},
            },
        }
    elif token_position == "last_token":
        token_position_specs = {
            "last_token": {"type": "index", "position": -1},
        }
    else:
        raise ValueError(
            f"Unknown token_position={token_position!r}. Expected 'A' or 'last_token'."
        )
    factories = build_token_position_factories(token_position_specs, template)
    token_positions = {}
    for name, factory in factories.items():
        token_positions[name] = factory(pipeline)
    return token_positions


def run_experiment(
    model_name: str,
    dataset_size: int = 128,
    batch_size: int = 32,
    device: str | None = None,
    output_dir: str | None = None,
    token_position: str = "A",
    verbose: bool = True,
):
    """
    Run the full capital alignment experiment: generate data, filter, run IIA, plot heatmap.
    """
    if output_dir is None:
        output_dir = Path(_SCRIPT_DIR) / "results"
    output_dir = Path(output_dir)
    datasets_dir = output_dir / "datasets"
    results_dir = output_dir / "results"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Config and causal model
    config = CapitalTaskConfig()
    causal_model = create_capital_model(config)

    if verbose:
        print("=" * 70)
        print("CAPITAL FACTUAL RECALL ALIGNMENT")
        print("=" * 70)
        print(f"\nModel: {model_name}")
        print(f"Template: {config.prompt_template!r}")
        print(f"Pairs: {len(config.capital_pairs)} (region, capital)")
        print(f"Dataset size: {dataset_size}, batch size: {batch_size}")
        print(f"Token position: {token_position}")
        print(f"Device: {device}")
        print()

    # Load model
    if verbose:
        print("Loading model...")
    pipeline = LMPipeline(
        model_name,
        max_new_tokens=10,
        device=device,
        max_length=64,
    )
    pipeline.tokenizer.padding_side = "left"
    num_layers = pipeline.model.config.num_hidden_layers
    if verbose:
        print(f"  Layers: {num_layers}")
        print()

    # Generate counterfactual dataset
    if verbose:
        print("Generating counterfactual dataset...")
    generator = lambda: next(capital_counterfactual_generator(config, causal_model))
    dataset = generate_counterfactual_samples(dataset_size, generator)
    if verbose:
        ex = dataset[0]
        print(f"  Generated {len(dataset)} pairs")
        print(f"  Example base: {ex['input']['raw_input']!r} -> {ex['input']['raw_output']!r}")
        print(f"  Example cf:   {ex['counterfactual_inputs'][0]['raw_input']!r} -> {ex['counterfactual_inputs'][0]['raw_output']!r}")
        print()

    # Filter: keep only examples where both base and counterfactual are correct
    if verbose:
        print("Filtering dataset (base and counterfactual must be correct)...")
    filtered_dataset = filter_dataset(
        dataset=dataset,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=batch_size,
        validate_counterfactuals=True,
    )
    keep_rate = len(filtered_dataset) / len(dataset) * 100 if dataset else 0
    if verbose:
        print(f"  Original: {len(dataset)}, Filtered: {len(filtered_dataset)} ({keep_rate:.1f}% kept)")
        print()

    if len(filtered_dataset) == 0:
        raise ValueError(
            "All examples were filtered out. The model may not perform well on this task. "
            "Try a larger model or check the prompt template."
        )

    # Save datasets
    filtered_path = datasets_dir / "filtered_capital_dataset.json"
    original_path = datasets_dir / "original_capital_dataset.json"
    save_counterfactual_examples(dataset, str(original_path))
    save_counterfactual_examples(filtered_dataset, str(filtered_path))
    if verbose:
        print(f"Saved datasets to {datasets_dir}")
        print()

    # Token position for A
    token_positions_dict = create_token_positions(
        pipeline, config, token_position=token_position
    )
    token_positions = list(token_positions_dict.values())

    # Layers: embedding (-1) + all transformer layers
    layers = [-1] + list(range(num_layers))

    # Build residual stream targets: one target per (layer, position_A)
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    if verbose:
        print(f"Running interchange interventions (IIA at position {token_position} across layers)...")
        print(f"  Layers: {len(layers)}, Position: {token_position}")
        print()

    # Run heatmap experiment: score = match to counterfactual's raw_output
    result = run_interchange_score_heatmap(
        causal_model=causal_model,
        interchange_targets=targets,
        dataset_path=str(filtered_path),
        pipeline=pipeline,
        target_variable_groups=[("raw_output",)],
        batch_size=batch_size,
        output_dir=str(results_dir),
        metric=checker,
        save_results=True,
        verbose=verbose,
    )

    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {results_dir}")
        print(f"Heatmaps: {results_dir / 'heatmaps'}")
        if "scores" in result:
            for var_group, scores in result["scores"].items():
                avg = sum(scores.values()) / len(scores) if scores else 0
                print(f"  {var_group}: avg IIA = {avg:.3f}")
        print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run capital factual recall alignment (IIA at position A, heatmap by layer)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=128,
        help="Number of counterfactual pairs to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for filtering and interventions",
    )
    parser.add_argument(
        "--token-position",
        type=str,
        default="A",
        choices=["A", "last_token"],
        help="Intervention token position: 'A' (last token of region name) or 'last_token' (end of prompt)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID (default: 0 if CUDA available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {Path(_SCRIPT_DIR) / 'results'})",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (sets HF_HOME)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir

    device = None
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"

    run_experiment(
        model_name=args.model,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
        token_position=args.token_position,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
