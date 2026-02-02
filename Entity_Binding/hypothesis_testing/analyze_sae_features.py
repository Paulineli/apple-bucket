#!/usr/bin/env python
"""
Analyze SAE Features Across Graph Clusters

This script:
1. Loads filtered input samples and partition results
2. Uses SAE from sae_lens to analyze activations at a given layer
3. Extracts activations from the last token
4. Finds features that fire in the first cluster but not in the second cluster (low IIA)
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Print diagnostic info about Python environment (helpful for debugging import issues)
if __name__ == "__main__" and len(sys.argv) > 1 and "--help" not in sys.argv:
    print(f"[DEBUG] Python executable: {sys.executable}", file=sys.stderr)
    print(f"[DEBUG] Python version: {sys.version}", file=sys.stderr)

# Add causalab to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..', 'causalab'))

from causalab.tasks.entity_binding.config import (
    EntityBindingTaskConfig,
    create_filling_liquids_config,
    create_sample_action_config,
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_entity_causal_model,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_position_builder import get_last_token_index, TokenPosition
from causalab.neural.LM_units import ResidualStream
from causalab.neural.pyvene_core.collect import collect_features
from causalab.causal.counterfactual_dataset import CounterfactualExample

# Try to import sae_lens
SAE_LENS_AVAILABLE = False
SAE = None
SparseAutoencoder = None

try:
    # First try the standard import
    from sae_lens import SAE
    # SparseAutoencoder might not always be available
    try:
        from sae_lens import SparseAutoencoder
    except ImportError:
        SparseAutoencoder = None
    SAE_LENS_AVAILABLE = True
except ImportError as e:
    import sys
    # Try importing the module directly first
    try:
        import sae_lens
        # Try to get SAE from the module
        if hasattr(sae_lens, 'SAE'):
            SAE = sae_lens.SAE
            SparseAutoencoder = getattr(sae_lens, 'SparseAutoencoder', None)
            SAE_LENS_AVAILABLE = True
            print("Successfully imported sae_lens using alternative method")
        else:
            # Try to find SAE class in submodules
            if hasattr(sae_lens, 'sae'):
                sae_module = sae_lens.sae
                if hasattr(sae_module, 'SAE'):
                    SAE = sae_module.SAE
                    SAE_LENS_AVAILABLE = True
                    print("Successfully imported SAE from sae_lens.sae")
    except Exception as e2:
        # If all imports fail, provide diagnostics
        print(f"Could not import sae_lens: {e}")
        print(f"Alternative import also failed: {e2}")
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {sys.version}")
        
        # Check if package is installed
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if 'sae-lens' in result.stdout or 'sae_lens' in result.stdout:
                print("Note: sae-lens appears to be installed, but import failed.")
                print("This might be a version compatibility issue or missing dependencies.")
                print("Try: pip install --upgrade sae-lens")
            else:
                print("Note: sae-lens does not appear to be installed in this environment.")
                print(f"Please install with: {sys.executable} -m pip install sae-lens")
        except Exception:
            pass


def register_qwen3_for_pyvene():
    """Register Qwen3 support for pyvene if needed."""
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
        )
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_type_to_dimension_mapping
    except Exception:
        pass

register_qwen3_for_pyvene()


def create_config(task_type: str, num_groups: int) -> EntityBindingTaskConfig:
    """Create task config."""
    if task_type == "filling_liquids":
        config = create_filling_liquids_config()
    else:
        config = create_sample_action_config()
    
    config.max_groups = num_groups
    config.prompt_prefix = "We will ask a question about the following sentences. Only return the answer, no other text.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"
    
    return config


def extract_activations_at_layer(
    pipeline: LMPipeline,
    samples: List[Dict],
    layer: int,
    batch_size: int = 32,
    task_type: str = "filling_liquids",
    num_groups: int = 6
) -> torch.Tensor:
    """
    Extract activations from the last token at the specified layer.
    
    Uses the collect_features function from the codebase to properly extract
    activations using pyvene.
    
    Args:
        pipeline: LMPipeline object
        samples: List of input samples (dicts with causal model variables)
        layer: Layer to extract activations from
        batch_size: Batch size for processing
        task_type: Task type for creating config
        num_groups: Number of groups for creating config
        
    Returns:
        Tensor of shape (n_samples, hidden_dim) containing activations
    """
    config = create_config(task_type, num_groups)
    causal_model = create_positional_entity_causal_model(config)
    
    # Convert samples to CounterfactualExample format (input only, no counterfactuals)
    dataset: List[CounterfactualExample] = []
    for sample in samples:
        trace = causal_model.new_trace(sample)
        dataset.append({
            "input": trace,
            "counterfactual_inputs": []
        })
    
    # Create token position for last token
    last_token_position = TokenPosition(
        lambda x: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )
    
    # Create ResidualStream unit for the specified layer
    hidden_size = pipeline.model.config.hidden_size
    residual_unit = ResidualStream(
        layer=layer,
        token_indices=last_token_position,
        featurizer=None,  # Identity featurizer
        shape=(hidden_size,),
        feature_indices=None,
        target_output=True,  # Get block_output (after the layer)
    )
    
    # Collect activations using the codebase function
    print("  Collecting activations using collect_features...")
    features_dict = collect_features(
        dataset=dataset,
        pipeline=pipeline,
        model_units=[residual_unit],
        batch_size=batch_size
    )
    
    # Extract the activations tensor
    unit_id = residual_unit.id
    if unit_id not in features_dict:
        raise ValueError(f"Could not find activations for unit {unit_id}")
    
    activations = features_dict[unit_id]
    
    return activations


def list_available_saes(model_name: str):
    """
    List available SAEs for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of available SAE information
    """
    available_saes = []
    
    if "gemma" in model_name.lower() and "2b" in model_name.lower():
        # For gemma-2-2b-it, only layer 12 is available
        available_saes.append({
            "release": "gemma-2b-it-res-jb",
            "sae_id": "blocks.12.hook_resid_post",
            "layer": 12,
            "note": "Only layer 12 has pretrained SAE available"
        })
    
    return available_saes


def load_sae(model_name: str, layer: int, device: str = "cuda:0", sae_path: str = None):
    """
    Load a pre-trained SAE for the specified model and layer.
    
    Args:
        model_name: Name of the model (e.g., "google/gemma-2-2b-it")
        layer: Layer number
        device: Device to load SAE on
        sae_path: Optional path to SAE (HuggingFace model ID, local path, or "release:sae_id" format)
        
    Returns:
        SAE object
    """
    if not SAE_LENS_AVAILABLE or SAE is None:
        import sys
        error_msg = (
            "sae_lens is not available. Please install with: pip install sae-lens\n"
            f"Current Python: {sys.executable}\n"
            "Make sure you're running the script with the correct Python interpreter "
            "that has sae-lens installed."
        )
        raise ImportError(error_msg)
    
    if sae_path:
        # Try loading from the specified path
        try:
            # Check if it's a local path
            if os.path.exists(sae_path):
                print(f"Loading SAE from local path: {sae_path}")
                sae = SAE.load_from_disk(sae_path, device=device)
            elif ":" in sae_path:
                # Format: "release:sae_id"
                release, sae_id = sae_path.split(":", 1)
                print(f"Loading SAE: release={release}, sae_id={sae_id}")
                try:
                    # Try new API format (if available)
                    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
                except TypeError:
                    # Try old API format (returns tuple)
                    sae, cfg, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            else:
                # Try as HuggingFace model ID
                print(f"Loading SAE from HuggingFace: {sae_path}")
                try:
                    # New API: SAE.from_pretrained(model_id, device=device)
                    sae = SAE.from_pretrained(sae_path, device=device)
                except Exception:
                    raise ValueError(f"Could not load SAE from {sae_path}. Try format 'release:sae_id'")
            print(f"Successfully loaded SAE")
            return sae
        except Exception as e:
            error_msg = f"Failed to load SAE from {sae_path}: {e}\n"
            # Check available SAEs and suggest
            available = list_available_saes(model_name)
            if available:
                error_msg += "\nAvailable SAEs for this model:\n"
                for sae_info in available:
                    error_msg += f"  --sae-path {sae_info['release']}:{sae_info['sae_id']} (layer {sae_info['layer']})\n"
            raise ValueError(error_msg)
    
    # Try to auto-detect SAE path for Gemma models
    sae_paths_to_try = []
    
    # For gemma models, try common patterns
    if "gemma" in model_name.lower() and "2b" in model_name.lower():
        # Correct release name is "gemma-2b-it-res-jb" (not "gemma-2-2b-it-res-jb")
        # Only layer 12 is available
        releases_to_try = [
            "gemma-2b-it-res-jb",  # Correct release name
        ]
        sae_ids_to_try = [
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.hook_resid_pre",
        ]
        
        for release in releases_to_try:
            for sae_id in sae_ids_to_try:
                sae_paths_to_try.append((release, sae_id))
    
    sae = None
    last_error = None
    for release, sae_id in sae_paths_to_try:
        try:
            print(f"Trying to load SAE: release={release}, sae_id={sae_id}")
            try:
                sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            except TypeError:
                # Old API returns tuple
                sae, cfg, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            print(f"Successfully loaded SAE")
            break
        except Exception as e:
            last_error = e
            print(f"Failed: {e}")
            continue
    
    if sae is None:
        error_msg = (
            f"Could not auto-detect SAE for {model_name} layer {layer}.\n"
            f"Last error: {last_error}\n\n"
        )
        # Check available SAEs and suggest
        available = list_available_saes(model_name)
        if available:
            error_msg += "Available SAEs for this model:\n"
            for sae_info in available:
                error_msg += f"  --sae-path {sae_info['release']}:{sae_info['sae_id']} (layer {sae_info['layer']})\n"
            if layer not in [s['layer'] for s in available]:
                error_msg += f"\nNote: No SAE available for layer {layer}. Available layers: {[s['layer'] for s in available]}\n"
        else:
            error_msg += (
                f"Please specify the correct SAE path using --sae-path.\n"
                f"Format: --sae-path <release>:<sae_id> or --sae-path <local_path>\n"
                f"Check available SAEs at: https://jbloomaus.github.io/SAELens/sae_table/\n"
            )
        raise ValueError(error_msg)
    
    return sae


def analyze_features_by_cluster(
    activations: torch.Tensor,
    labels: List[int],
    sae: SAE,
    cluster_0_id: int = 0,
    cluster_1_id: int = 1,
    threshold: float = 0.0
) -> Dict:
    """
    Analyze which SAE features fire in cluster 0 but not in cluster 1.
    
    Args:
        activations: Tensor of shape (n_samples, hidden_dim) with raw activations
        labels: List of cluster labels for each sample
        sae: SAE object to encode activations
        cluster_0_id: ID of first cluster (high IIA)
        cluster_1_id: ID of second cluster (low IIA)
        threshold: Threshold for considering a feature as "firing"
        
    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy for easier indexing
    labels = np.array(labels)
    
    # Get cluster masks
    cluster_0_mask = labels == cluster_0_id
    cluster_1_mask = labels == cluster_1_id
    
    cluster_0_indices = np.where(cluster_0_mask)[0]
    cluster_1_indices = np.where(cluster_1_mask)[0]
    
    print(f"Cluster 0 (high IIA): {len(cluster_0_indices)} samples")
    print(f"Cluster 1 (low IIA): {len(cluster_1_indices)} samples")
    
    # Encode activations through SAE
    print("Encoding activations through SAE...")
    with torch.no_grad():
        # SAE.encode returns feature activations
        # activations should be on the same device as SAE
        device = next(sae.parameters()).device
        activations = activations.to(device)
        
        # Encode all activations
        # sae_lens API: sae.encode() returns a SparseAutoencoderOutput object
        # which has feature_acts attribute
        sae_output = sae.encode(activations)
        
        # Handle different return types
        if hasattr(sae_output, 'feature_acts'):
            feature_activations = sae_output.feature_acts  # Shape: (n_samples, n_features)
        elif isinstance(sae_output, torch.Tensor):
            feature_activations = sae_output
        elif isinstance(sae_output, tuple):
            # Some versions return (feature_acts, ...)
            feature_activations = sae_output[0]
        else:
            raise ValueError(f"Unexpected SAE output type: {type(sae_output)}")
    
    # Analyze features for each cluster
    cluster_0_features = feature_activations[cluster_0_indices]  # (n_cluster0, n_features)
    cluster_1_features = feature_activations[cluster_1_indices]  # (n_cluster1, n_features)
    
    # Compute mean activation per feature for each cluster
    cluster_0_mean = cluster_0_features.mean(dim=0).cpu().numpy()  # (n_features,)
    cluster_1_mean = cluster_1_features.mean(dim=0).cpu().numpy()  # (n_features,)
    
    # Compute fraction of samples where feature fires (above threshold)
    cluster_0_firing = (cluster_0_features > threshold).float().mean(dim=0).cpu().numpy()
    cluster_1_firing = (cluster_1_features > threshold).float().mean(dim=0).cpu().numpy()
    
    # Find features that fire in cluster 0 but not in cluster 1
    # Criteria: high firing rate in cluster 0, low firing rate in cluster 1
    firing_diff = cluster_0_firing - cluster_1_firing
    
    # Features that fire more in cluster 0
    cluster_0_specific = np.where(
        (cluster_0_firing > 0.5) & (cluster_1_firing < 0.3)
    )[0]
    
    # Sort by difference
    sorted_indices = np.argsort(firing_diff)[::-1]
    
    # Get top features
    top_n = min(100, len(sorted_indices))
    top_features = sorted_indices[:top_n]
    
    results = {
        "cluster_0_id": cluster_0_id,
        "cluster_1_id": cluster_1_id,
        "cluster_0_size": len(cluster_0_indices),
        "cluster_1_size": len(cluster_1_indices),
        "n_features": feature_activations.shape[1],
        "cluster_0_specific_features": cluster_0_specific.tolist(),
        "top_features_by_difference": top_features.tolist(),
        "firing_diff": firing_diff.tolist(),
        "cluster_0_mean_activations": cluster_0_mean.tolist(),
        "cluster_1_mean_activations": cluster_1_mean.tolist(),
        "cluster_0_firing_rates": cluster_0_firing.tolist(),
        "cluster_1_firing_rates": cluster_1_firing.tolist(),
    }
    
    return results


def train_cluster_classifier(
    feature_activations: torch.Tensor,
    labels: List[int],
    top_feature_indices: List[int],
    cluster_0_id: int,
    cluster_1_id: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Train a classifier to predict cluster membership using top features.
    
    Args:
        feature_activations: Tensor of shape (n_samples, n_features) with SAE feature activations
        labels: List of cluster labels for each sample
        top_feature_indices: List of feature indices to use for classification
        cluster_0_id: ID of first cluster
        cluster_1_id: ID of second cluster
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with classification results including test accuracy
    """
    # Convert to numpy
    labels = np.array(labels)
    feature_activations = feature_activations.cpu().numpy()
    
    # Filter to only samples from the two clusters
    cluster_mask = (labels == cluster_0_id) | (labels == cluster_1_id)
    X = feature_activations[cluster_mask]
    y = labels[cluster_mask]
    
    # Convert cluster IDs to binary labels (0 for cluster_0_id, 1 for cluster_1_id)
    y_binary = (y == cluster_1_id).astype(int)
    
    # Extract only the top features
    X_top_features = X[:, top_feature_indices]
    
    print(f"\nTraining classifier with top {len(top_feature_indices)} features...")
    print(f"  Features used: {top_feature_indices}")
    print(f"  Total samples: {len(X_top_features)}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_top_features,
        y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train logistic regression classifier
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Also evaluate on training set
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Get classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[f"Cluster {cluster_0_id}", f"Cluster {cluster_1_id}"],
        output_dict=True
    )
    
    results = {
        "top_feature_indices": top_feature_indices,
        "n_features_used": len(top_feature_indices),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "classification_report": report,
        "cluster_0_id": cluster_0_id,
        "cluster_1_id": cluster_1_id,
        "classifier": classifier,  # Include the trained classifier
    }
    
    return results


def train_cluster_classifier_full_l1(
    feature_activations: torch.Tensor,
    labels: List[int],
    cluster_0_id: int,
    cluster_1_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 0.01,
    max_iter: int = 2000,
) -> Dict:
    """
    Train a classifier to predict cluster membership using the full SAE feature vector
    with L1-regularized logistic regression.
    
    This is intended for very high-dimensional SAE features, so we use a strong
    L1 penalty (small C) to induce sparsity.
    """
    # Convert to numpy
    labels = np.array(labels)
    feature_activations = feature_activations.cpu().numpy()
    
    # Filter to only samples from the two clusters
    cluster_mask = (labels == cluster_0_id) | (labels == cluster_1_id)
    X = feature_activations[cluster_mask]
    y = labels[cluster_mask]
    
    # Convert cluster IDs to binary labels (0 for cluster_0_id, 1 for cluster_1_id)
    y_binary = (y == cluster_1_id).astype(int)
    
    print(f"\nTraining L1 logistic regression classifier on full feature vector...")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Total samples: {len(X)}")
    print(f"  Using C={C} (smaller C = stronger regularization)")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # L1-regularized logistic regression on high-dimensional data
    classifier = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Also evaluate on training set
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Get classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[f"Cluster {cluster_0_id}", f"Cluster {cluster_1_id}"],
        output_dict=True
    )
    
    # Measure sparsity of learned classifier
    # coef_ has shape (1, n_features) for binary logistic regression
    n_features_used = classifier.coef_.shape[1]
    n_nonzero = int(np.count_nonzero(classifier.coef_))
    
    results = {
        "top_feature_indices": None,  # Full vector
        "n_features_used": n_features_used,
        "n_nonzero_features": n_nonzero,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "classification_report": report,
        "cluster_0_id": cluster_0_id,
        "cluster_1_id": cluster_1_id,
        "classifier": classifier,
        "mode": "full-l1",
        "C": C,
        "max_iter": max_iter,
    }
    
    return results


def save_classifier(
    classifier: LogisticRegression,
    top_feature_indices: List[int],
    cluster_0_id: int,
    cluster_1_id: int,
    save_path: str
) -> None:
    """
    Save classifier and metadata to disk.
    
    Args:
        classifier: Trained LogisticRegression classifier
        top_feature_indices: List of feature indices used by classifier
        cluster_0_id: ID of first cluster
        cluster_1_id: ID of second cluster
        save_path: Path to save classifier (will save as .pkl file)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save classifier
    classifier_path = save_path.with_suffix('.pkl')
    joblib.dump(classifier, classifier_path)
    
    # Save metadata
    metadata = {
        "top_feature_indices": top_feature_indices if top_feature_indices is not None else "all",
        "cluster_0_id": cluster_0_id,
        "cluster_1_id": cluster_1_id,
        "n_features": len(top_feature_indices) if top_feature_indices is not None else int(classifier.coef_.shape[1]),
    }
    metadata_path = save_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Classifier saved to: {classifier_path}")
    print(f"Metadata saved to: {metadata_path}")


def load_classifier(classifier_path: str) -> Tuple[LogisticRegression, Dict]:
    """
    Load classifier and metadata from disk.
    
    Args:
        classifier_path: Path to classifier .pkl file
        
    Returns:
        Tuple of (classifier, metadata_dict)
    """
    classifier_path = Path(classifier_path)
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
    
    # Load classifier
    classifier = joblib.load(classifier_path)
    
    # Load metadata
    metadata_path = classifier_path.with_suffix('.json')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return classifier, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE features across graph clusters"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to filtered_input_samples pkl file"
    )
    parser.add_argument(
        "--partition-results-path",
        type=str,
        required=True,
        help="Path to partition_results json file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Model name"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="Path to SAE (HuggingFace model ID, local path, or 'release:sae_id' format). "
             "For gemma-2-2b-it, use: --sae-path gemma-2b-it-res-jb:blocks.12.hook_resid_post "
             "(only layer 12 has pretrained SAE). If not specified, will try common patterns."
    )
    parser.add_argument(
        "--list-saes",
        action="store_true",
        help="List available SAEs for the specified model and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: same directory as partition results)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for considering a feature as 'firing' (default: 0.0)"
    )
    parser.add_argument(
        "--save-classifier",
        type=str,
        default=None,
        help="Path to save the trained classifier (default: don't save)"
    )
    parser.add_argument(
        "--classifier-mode",
        type=str,
        choices=["top-k", "full-l1"],
        default="top-k",
        help="How to train the cluster classifier: "
             "'top-k' uses the top 10 SAE features by firing difference; "
             "'full-l1' uses the full SAE feature vector with L1-regularized logistic regression."
    )
    parser.add_argument(
        "--full-l1-C",
        type=float,
        default=0.01,
        help="Inverse regularization strength C for full-vector L1 logistic regression "
             "(smaller C means stronger L1 regularization; default=0.01)."
    )
    
    args = parser.parse_args()
    
    # If --list-saes flag is set, list available SAEs and exit
    if args.list_saes:
        print("=" * 70)
        print("Available SAEs for", args.model)
        print("=" * 70)
        available = list_available_saes(args.model)
        if available:
            for sae_info in available:
                print(f"\nRelease: {sae_info['release']}")
                print(f"SAE ID: {sae_info['sae_id']}")
                print(f"Layer: {sae_info['layer']}")
                if 'note' in sae_info:
                    print(f"Note: {sae_info['note']}")
                print(f"\nUsage: --sae-path {sae_info['release']}:{sae_info['sae_id']}")
        else:
            print("No pretrained SAEs found for this model.")
            print("Check available SAEs at: https://jbloomaus.github.io/SAELens/sae_table/")
        print("=" * 70)
        return 0
    
    # Check SAE availability early and provide helpful diagnostics
    if not SAE_LENS_AVAILABLE:
        import sys
        print("=" * 70)
        print("ERROR: sae_lens is not available")
        print("=" * 70)
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {sys.version}")
        print("\nTroubleshooting:")
        print("1. Make sure your virtual environment is activated")
        print("2. Verify installation: pip show sae-lens")
        print("3. Try reinstalling: pip install --upgrade sae-lens")
        print("4. Check if you're using the correct Python interpreter")
        print("\nTo check which Python you're using, run:")
        print(f"  {sys.executable} -m pip list | grep sae")
        print("=" * 70)
        sys.exit(1)
    
    # Setup device
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    
    print("=" * 70)
    print("SAE Feature Analysis Across Clusters")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Partition results: {args.partition_results_path}")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"  Loaded {len(samples)} samples")
    
    # Load partition results
    print("Loading partition results...")
    partition_path = Path(args.partition_results_path)
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition results file not found: {partition_path}")
    
    with open(partition_path, 'r') as f:
        partition_results = json.load(f)
    
    labels = partition_results["labels"]
    iia_by_cluster = partition_results["iia_by_cluster"]
    
    print(f"  Loaded labels for {len(labels)} samples")
    print(f"  IIA by cluster: {iia_by_cluster}")
    
    # Identify clusters (cluster 0 = high IIA, cluster 1 = low IIA)
    cluster_0_id = 0
    cluster_1_id = 1
    
    # Verify we have the right clusters
    if iia_by_cluster.get(str(cluster_0_id), 0) < iia_by_cluster.get(str(cluster_1_id), 1):
        # Swap if needed
        cluster_0_id, cluster_1_id = cluster_1_id, cluster_0_id
        print(f"  Swapped clusters: cluster {cluster_0_id} has high IIA, cluster {cluster_1_id} has low IIA")
    else:
        print(f"  Cluster {cluster_0_id} has high IIA, cluster {cluster_1_id} has low IIA")
    
    # Load model
    print(f"\nLoading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=5,
        device=device,
        max_length=200,
    )
    pipeline.tokenizer.padding_side = "left"
    print("  Model loaded")
    
    # Load SAE
    print(f"\nLoading SAE for layer {args.layer}...")
    sae = load_sae(args.model, args.layer, device=device, sae_path=args.sae_path)
    
    # Get number of features
    if hasattr(sae, 'cfg') and hasattr(sae.cfg, 'd_sae'):
        n_features = sae.cfg.d_sae
    elif hasattr(sae, 'W_enc'):
        n_features = sae.W_enc.shape[0]  # Number of features
    else:
        n_features = "unknown"
    
    print(f"  SAE has {n_features} features")
    
    # Get task type and num_groups from partition results
    task_type = partition_results.get("task", "filling_liquids")
    # Infer num_groups from sample_size if not directly available
    # We can also get it from the config used to generate samples
    num_groups = partition_results.get("num_groups", 6)
    
    # Extract activations
    print(f"\nExtracting activations from layer {args.layer} (last token)...")
    activations = extract_activations_at_layer(
        pipeline,
        samples,
        args.layer,
        batch_size=args.batch_size,
        task_type=task_type,
        num_groups=num_groups  # Default, can be inferred from dataset
    )
    print(f"  Extracted activations shape: {activations.shape}")
    
    # Move activations to device if needed
    if activations.device != device:
        activations = activations.to(device)
    
    # Analyze features
    print(f"\nAnalyzing SAE features across clusters...")
    results = analyze_features_by_cluster(
        activations,
        labels,
        sae,
        cluster_0_id=cluster_0_id,
        cluster_1_id=cluster_1_id,
        threshold=args.threshold
    )
    
    print(f"\nResults:")
    print(f"  Total features: {results['n_features']}")
    print(f"  Cluster 0 specific features: {len(results['cluster_0_specific_features'])}")
    print(f"  Top 10 features by firing difference:")
    for i, feat_idx in enumerate(results['top_features_by_difference'][:10]):
        firing_diff = results['firing_diff'][feat_idx]
        cluster_0_firing = results['cluster_0_firing_rates'][feat_idx]
        cluster_1_firing = results['cluster_1_firing_rates'][feat_idx]
        print(f"    Feature {feat_idx}: diff={firing_diff:.3f}, "
              f"cluster_0={cluster_0_firing:.3f}, cluster_1={cluster_1_firing:.3f}")
    
    # Train classifier using top 10 features
    print(f"\n" + "=" * 70)
    print("Training Cluster Classifier")
    print("=" * 70)
    
    # Re-encode activations to get SAE feature activations (needed for classifier)
    print("Encoding activations for classifier...")
    with torch.no_grad():
        device = next(sae.parameters()).device
        activations_for_sae = activations.to(device)
        sae_output = sae.encode(activations_for_sae)
        
        # Handle different return types
        if hasattr(sae_output, 'feature_acts'):
            feature_activations = sae_output.feature_acts
        elif isinstance(sae_output, torch.Tensor):
            feature_activations = sae_output
        elif isinstance(sae_output, tuple):
            feature_activations = sae_output[0]
        else:
            raise ValueError(f"Unexpected SAE output type: {type(sae_output)}")
    
    # Train classifier according to selected mode
    if args.classifier_mode == "top-k":
        # Get top 10 features
        top_10_features = results['top_features_by_difference'][:10]
        classifier_results = train_cluster_classifier(
            feature_activations,
            labels,
            top_10_features,
            cluster_0_id=cluster_0_id,
            cluster_1_id=cluster_1_id
        )
    else:
        classifier_results = train_cluster_classifier_full_l1(
            feature_activations,
            labels,
            cluster_0_id=cluster_0_id,
            cluster_1_id=cluster_1_id,
            C=args.full_l1_C,
        )
    
    print(f"\nClassifier Results:")
    if args.classifier_mode == "top-k":
        print(f"  Mode: top-k (k={classifier_results['n_features_used']})")
        print(f"  Features used: {classifier_results['top_feature_indices']}")
    else:
        print(f"  Mode: full-l1")
        print(f"  Total features: {classifier_results['n_features_used']}")
        print(f"  Non-zero coefficients: {classifier_results['n_nonzero_features']}")
    print(f"  Train accuracy: {classifier_results['train_accuracy']:.4f}")
    print(f"  Test accuracy: {classifier_results['test_accuracy']:.4f}")
    print(f"\n  Classification Report:")
    print(f"    Precision (Cluster {cluster_0_id}): {classifier_results['classification_report'][f'Cluster {cluster_0_id}']['precision']:.4f}")
    print(f"    Recall (Cluster {cluster_0_id}): {classifier_results['classification_report'][f'Cluster {cluster_0_id}']['recall']:.4f}")
    print(f"    Precision (Cluster {cluster_1_id}): {classifier_results['classification_report'][f'Cluster {cluster_1_id}']['precision']:.4f}")
    print(f"    Recall (Cluster {cluster_1_id}): {classifier_results['classification_report'][f'Cluster {cluster_1_id}']['recall']:.4f}")
    
    # Add classifier results to main results (without the classifier object for JSON serialization)
    classifier_results_for_json = {k: v for k, v in classifier_results.items() if k != "classifier"}
    results["classifier_results"] = classifier_results_for_json
    
    # Save classifier if requested
    if args.save_classifier:
        save_classifier(
            classifier_results["classifier"],
            classifier_results["top_feature_indices"],
            cluster_0_id,
            cluster_1_id,
            args.save_classifier
        )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        # If output_path is a directory, create a filename inside it
        if output_path.is_dir():
            output_path = output_path / f"sae_analysis_layer_{args.layer}.json"
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = partition_path.parent / f"sae_analysis_layer_{args.layer}.json"
    
    # Add metadata
    results["metadata"] = {
        "model": args.model,
        "layer": args.layer,
        "dataset_path": str(dataset_path),
        "partition_results_path": str(partition_path),
        "threshold": args.threshold,
        "sae_path": args.sae_path if args.sae_path else "auto-detected",
        "non_zero_features": classifier_results["n_nonzero_features"],
        "classifier_mode": args.classifier_mode,
        "Precision_Cluster_0": classifier_results["classification_report"][f"Cluster {cluster_0_id}"]["precision"],
        "Recall_Cluster_0": classifier_results["classification_report"][f"Cluster {cluster_0_id}"]["recall"],
        "Precision_Cluster_1": classifier_results["classification_report"][f"Cluster {cluster_1_id}"]["precision"],
        "Recall_Cluster_1": classifier_results["classification_report"][f"Cluster {cluster_1_id}"]["recall"],
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
