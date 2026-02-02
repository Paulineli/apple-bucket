import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, Optional, Union
import warnings
import json
import os
import argparse
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')


def learn_hyperplane(
    df: pd.DataFrame,
    label_col: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    normalize: bool = True,
    model_type: str = 'logistic',
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    learning_rate_init: float = 0.001,
    alpha: float = 0.0001,
    class_weight: Optional[Union[str, dict]] = 'balanced'
) -> Tuple[Union[LogisticRegression, MLPClassifier], StandardScaler, dict]:
    """
    Learn a hyperplane to separate binary labeled data (0 or 1).
    
    This function can use either Logistic Regression or a Neural Network to find an optimal
    decision boundary (hyperplane) that separates the two classes. Logistic regression provides
    a linear boundary and is more interpretable, while neural networks can learn non-linear
    boundaries and may achieve better performance on complex datasets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and labels
    label_col : str, default='label'
        Name of the column containing binary labels (0 or 1)
    test_size : float, default=0.2
        Proportion of data to use for testing (between 0 and 1)
    random_state : int, default=42
        Random seed for reproducibility
    C : float, default=1.0
        Inverse of regularization strength for logistic regression. Smaller values = stronger regularization
    max_iter : int, default=1000
        Maximum number of iterations for the solver
    normalize : bool, default=True
        Whether to standardize features (recommended for numerical stability)
    model_type : str, default='logistic'
        Type of model to use: 'logistic' for Logistic Regression or 'neural_net' for Neural Network
    hidden_layer_sizes : Tuple[int, ...], default=(100,)
        Hidden layer sizes for neural network. Example: (100,) for single layer, (100, 50) for two layers
    learning_rate_init : float, default=0.001
        Initial learning rate for neural network optimizer
    alpha : float, default=0.0001
        L2 regularization parameter for neural network
    class_weight : str or dict, default='balanced'
        Class weight for both logistic regression and neural networks. Use 'balanced' to 
        automatically adjust weights inversely proportional to class frequencies, or None 
        for uniform weights. Helps handle imbalanced datasets.
    
    Returns:
    --------
    model : LogisticRegression or MLPClassifier
        Trained model (type depends on model_type parameter)
    scaler : StandardScaler or None
        Feature scaler (None if normalize=False)
    results : dict
        Dictionary containing:
        - 'train_accuracy': Training set accuracy
        - 'test_accuracy': Test set accuracy (computed without balancing/class weights, standard unweighted accuracy)
        - 'coefficients': Hyperplane coefficients (weights) - only for logistic regression
        - 'intercept': Hyperplane intercept (bias) - only for logistic regression
        - 'n_features': Number of features
        - 'n_samples': Number of samples
        - 'class_distribution': Count of each class
        - 'model_type': Type of model used ('logistic' or 'neural_net')
    
    Example:
    --------
    """
    
    # Validate input
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")
    
    # Check for binary labels
    unique_labels = df[label_col].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")
    
    # Separate features and labels
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    
    # Get dataset info
    n_samples, n_features = X.shape
    class_counts = pd.Series(y).value_counts().to_dict()
    
    print(f"Dataset info:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: {class_counts}")
    
    # Split data
    # Check if stratification is possible (need at least 2 samples per class in each split)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    min_class_count = min(label_counts)
    min_samples_per_split = max(2, int(min_class_count * min(test_size, 1 - test_size)))
    
    # Only use stratify if we have enough samples per class in each split
    # sklearn requires at least 2 samples per class in each split for stratification
    use_stratify = y if (min_class_count >= 2 and min_samples_per_split >= 2) else None
    if use_stratify is None:
        print(f"  Warning: Cannot use stratify - minimum class count is {min_class_count}")
        print(f"  Minimum samples per split needed: {min_samples_per_split}")
        print(f"  Proceeding without stratification")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=use_stratify
    )
    
    # Print train/test split info
    train_class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    test_class_counts = dict(zip(*np.unique(y_test, return_counts=True)))
    print(f"\nTrain/test split info (before balancing):")
    print(f"  Train set: {len(y_train)} samples, class distribution: {train_class_counts}")
    print(f"  Test set: {len(y_test)} samples, class distribution: {test_class_counts}")
    
    # Balance test set to have equal number of class 0 and class 1 samples
    test_class_0_indices = np.where(y_test == 0)[0]
    test_class_1_indices = np.where(y_test == 1)[0]
    n_class_0 = len(test_class_0_indices)
    n_class_1 = len(test_class_1_indices)
    
    if n_class_0 != n_class_1:
        # Find the minority class and target size
        min_count = min(n_class_0, n_class_1)
        
        # Downsample the majority class to match the minority class
        if n_class_0 > n_class_1:
            # Class 0 is majority, downsample it
            test_class_0_indices_balanced = resample(
                test_class_0_indices,
                replace=False,
                n_samples=min_count,
                random_state=random_state
            )
            test_class_1_indices_balanced = test_class_1_indices
        else:
            # Class 1 is majority, downsample it
            test_class_1_indices_balanced = resample(
                test_class_1_indices,
                replace=False,
                n_samples=min_count,
                random_state=random_state
            )
            test_class_0_indices_balanced = test_class_0_indices
        
        # Combine balanced indices
        test_balanced_indices = np.concatenate([test_class_0_indices_balanced, test_class_1_indices_balanced])
        
        # Update test set with balanced samples
        X_test_balanced = X_test[test_balanced_indices]
        y_test_balanced = y_test[test_balanced_indices]
        
        # Shuffle to mix classes
        rng = np.random.RandomState(random_state)
        shuffle_indices = rng.permutation(len(y_test_balanced))
        X_test = X_test_balanced[shuffle_indices]
        y_test = y_test_balanced[shuffle_indices]
        
        print(f"\nTest set balanced:")
        balanced_test_class_counts = dict(zip(*np.unique(y_test, return_counts=True)))
        print(f"  Test set: {len(y_test)} samples, class distribution: {balanced_test_class_counts}")
    else:
        print(f"\nTest set is already balanced: {n_class_0} samples per class")
    
    # Normalize features if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train model based on model_type
    if model_type == 'logistic':
        class_weight_str = class_weight if isinstance(class_weight, str) else 'custom'
        print(f"\nTraining logistic regression (C={C}, max_iter={max_iter}, class_weight={class_weight_str})...")
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',  # Efficient for medium datasets
            penalty='l2',
            class_weight=class_weight
        )
        model.fit(X_train, y_train)
        
        # Extract hyperplane parameters (only for logistic regression)
        coefficients = model.coef_[0]  # Shape: (n_features,)
        intercept = model.intercept_[0]  # Scalar
        
    elif model_type == 'neural_net':
        class_weight_str = class_weight if isinstance(class_weight, str) else 'custom'
        print(f"\nTraining neural network (hidden_layers={hidden_layer_sizes}, max_iter={max_iter}, class_weight={class_weight_str})...")

        # Define a simple feed-forward neural network in PyTorch
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...]):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev_dim, h))
                    layers.append(nn.ReLU())
                    prev_dim = h
                # Single output logit for binary classification
                layers.append(nn.Linear(prev_dim, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        # Use at most a single GPU (cuda:0) if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_dim = X_train.shape[1]
        torch_model = SimpleMLP(input_dim, hidden_layer_sizes).to(device)

        # Set up optimizer
        optimizer = torch.optim.Adam(
            torch_model.parameters(),
            lr=learning_rate_init,
            weight_decay=alpha  # L2 regularization
        )

        # Compute class weights for loss instead of resampling
        pos_weight_tensor = None
        y_train_np = y_train.astype(np.float32)
        unique_classes, class_counts_arr = np.unique(y_train_np, return_counts=True)

        if class_weight is not None:
            if class_weight == 'balanced':
                # Standard "balanced" strategy: inverse frequency weighting
                # For BCEWithLogitsLoss, pos_weight = weight_pos / weight_neg
                # Here we approximate using counts: (N_neg / N_pos)
                n_pos = class_counts_arr[unique_classes == 1][0] if 1 in unique_classes else 1.0
                n_neg = class_counts_arr[unique_classes == 0][0] if 0 in unique_classes else 1.0
                if n_pos > 0:
                    pos_weight_value = float(n_neg / n_pos)
                else:
                    pos_weight_value = 1.0
                pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                print(f"  Using balanced loss with pos_weight={pos_weight_value:.4f}")
            elif isinstance(class_weight, dict):
                # Map dict weights to a pos_weight for BCE
                w_pos = float(class_weight.get(1, 1.0))
                w_neg = float(class_weight.get(0, 1.0))
                if w_neg <= 0:
                    w_neg = 1.0
                pos_weight_value = w_pos / w_neg
                pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                print(f"  Using custom class_weight with pos_weight={pos_weight_value:.4f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Prepare tensors
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_train_tensor = torch.from_numpy(y_train_np.reshape(-1, 1)).to(device)

        torch_model.train()
        for epoch in range(max_iter):
            optimizer.zero_grad()
            logits = torch_model(X_train_tensor)
            loss = criterion(logits, y_train_tensor)
            loss.backward()
            optimizer.step()

        model = torch_model

        # Extract hyperplane approximation from final linear layer
        last_layer = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_layer = m
        if last_layer is not None:
            w = last_layer.weight.detach().cpu().numpy().reshape(-1)
            b = last_layer.bias.detach().cpu().numpy().reshape(-1)
            coefficients = w
            intercept = b[0] if b.size > 0 else 0.0
        else:
            coefficients = None
            intercept = None
            
    else:
        raise ValueError(f"model_type must be 'logistic' or 'neural_net', got '{model_type}'")
    
    # Evaluate model
    # Note: Test accuracy is computed WITHOUT balancing/class weights (standard unweighted accuracy)
    # For torch neural networks, compute accuracy manually
    if isinstance(model, (LogisticRegression, MLPClassifier)):
        train_accuracy = model.score(X_train, y_train)
        # Test accuracy computed without balancing (standard unweighted accuracy)
        test_accuracy = model.score(X_test, y_test)
        # Get test predictions for distribution analysis
        test_pred = model.predict(X_test)
    else:
        # Torch model branch (single device, same as used in training)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_train_tensor_eval = torch.from_numpy(X_train.astype(np.float32)).to(device)
            X_test_tensor_eval = torch.from_numpy(X_test.astype(np.float32)).to(device)
            train_logits = model(X_train_tensor_eval).cpu().numpy().reshape(-1)
            test_logits = model(X_test_tensor_eval).cpu().numpy().reshape(-1)
            train_pred = (1 / (1 + np.exp(-train_logits)) >= 0.5).astype(int)
            test_pred = (1 / (1 + np.exp(-test_logits)) >= 0.5).astype(int)
            train_accuracy = (train_pred == y_train).mean()
            # Test accuracy computed without balancing (standard unweighted accuracy)
            test_accuracy = (test_pred == y_test).mean()
        # Restore train mode for potential further use
        model.train()
        # Skip resampled accuracy since we no longer resample for torch models
        # (class imbalance is handled via the loss function)
    
    # Compute prediction distribution on test set
    test_pred_counts = dict(zip(*np.unique(test_pred, return_counts=True)))
    test_pred_distribution = {
        pred_class: {
            'count': int(count),
            'percentage': float(count / len(test_pred) * 100)
        }
        for pred_class, count in test_pred_counts.items()
    }
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy (unweighted): {test_accuracy:.4f}")
    print(f"\nTest set prediction distribution:")
    for pred_class in sorted(test_pred_distribution.keys()):
        dist = test_pred_distribution[pred_class]
        print(f"  Class {pred_class}: {dist['count']} ({dist['percentage']:.2f}%)")
    
    # Print hyperplane equation (only for logistic regression or if available)
    if model_type == 'logistic':
        print(f"\nHyperplane equation:")
        print(f"  Intercept (bias): {intercept:.4f}")
        print(f"  Coefficients (weights): {coefficients}")
    else:
        print(f"\nNote: Neural network with multiple layers doesn't have a simple hyperplane representation.")
    
    # Prepare results dictionary
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'coefficients': coefficients,
        'intercept': intercept,
        'n_features': n_features,
        'n_samples': n_samples,
        'class_distribution': class_counts,
        'test_prediction_distribution': test_pred_distribution,
        'feature_names': df.drop(columns=[label_col]).columns.tolist(),
        'model_type': model_type
    }
    
    return model, scaler, results


def predict_with_hyperplane(
    model: Union[LogisticRegression, MLPClassifier],
    X_new: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Make predictions on new data using the learned hyperplane.
    
    Parameters:
    -----------
    model : LogisticRegression or MLPClassifier
        Trained model from learn_hyperplane()
    X_new : pd.DataFrame
        New data to predict (features only, no label column)
    scaler : StandardScaler or None
        Scaler used during training (if any)
    
    Returns:
    --------
    predictions : np.ndarray
        Predicted labels (0 or 1)
    """
    X = X_new.values if isinstance(X_new, pd.DataFrame) else X_new
    
    if scaler is not None:
        X = scaler.transform(X)

    # sklearn models
    if hasattr(model, "predict"):
        return model.predict(X)

    # Torch model: manual prediction (single device)
    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            logits = model(X_tensor).cpu().numpy().reshape(-1)
            proba = 1 / (1 + np.exp(-logits))
            preds = (proba >= 0.5).astype(int)
        model.train()
        return preds

    raise TypeError("Unsupported model type for prediction.")


def get_decision_scores(
    model: Union[LogisticRegression, MLPClassifier],
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Get the decision function scores (distance from hyperplane).
    
    Positive scores indicate class 1, negative scores indicate class 0.
    The magnitude indicates confidence.
    
    Parameters:
    -----------
    model : LogisticRegression or MLPClassifier
        Trained model from learn_hyperplane()
    X : pd.DataFrame
        Data to score (features only)
    scaler : StandardScaler or None
        Scaler used during training (if any)
    
    Returns:
    --------
    scores : np.ndarray
        Decision function scores
    """
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    if scaler is not None:
        X_array = scaler.transform(X_array)
    
    # sklearn MLPClassifier: use predict_proba
    if isinstance(model, MLPClassifier):
        proba = model.predict_proba(X_array)
        if proba.shape[1] == 2:
            scores = np.log(proba[:, 1] / (proba[:, 0] + 1e-10))
        else:
            scores = proba[:, 1] - 0.5
        return scores

    # LogisticRegression: use decision_function if available
    if hasattr(model, "decision_function"):
        return model.decision_function(X_array)

    # Torch model: return raw logits as decision scores (single device)
    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(np.asarray(X_array, dtype=np.float32)).to(device)
            logits = model(X_tensor).cpu().numpy().reshape(-1)
        model.train()
        return logits

    raise TypeError("Unsupported model type for decision scores.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Train hyperplane classifier on causal abstraction dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['logistic', 'neural_net'],
        default='logistic',
        help='Classification method: logistic regression or neural network'
    )
    parser.add_argument(
        '--features', '-f',
        type=str,
        choices=['pqr', 'activation'],
        default='pqr',
        help='Feature set: pqr (p,q,r,ps,qs,rs) or activation (activation_base + activation_source)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (between 0 and 1)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: test_results/analysis_datasets_boundless_or_model_1.json)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the model'
    )
    parser.add_argument(
        '--hidden-layers',
        type=int,
        nargs='+',
        default=[100],
        help='Hidden layer sizes for neural network (e.g., --hidden-layers 100 50)'
    )
    parser.add_argument(
        '--class-weight',
        type=str,
        default='balanced',
        choices=['balanced', 'none'],
        help='Class weight for both logistic regression and neural networks: balanced (default) or none. Helps handle imbalanced datasets.'
    )
    
    args = parser.parse_args()
    
    # Load dataset from JSON file
    print("=" * 60)
    print("Hyperplane Classifier - Dataset Analysis")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Features: {args.features}")
    print("=" * 60)
    
    if args.dataset:
        json_path = args.dataset
    else:
        json_path = os.path.join(
            os.path.dirname(__file__),
            'test_results',
            'analysis_datasets_boundless_or_model_1.json'
        )
    
    print(f"\nLoading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract all features and labels from all entries
    features_option1 = []  # p, q, r, ps, qs, rs
    features_option2 = []  # activation_base + activation_source (flattened)
    labels_list = []
    
    # Iterate through all entries in the dataset
    for op_name, op_data in data.items():
        for entry_key, entry in op_data.items():
            if 'features' in entry and 'labels' in entry:
                for feat_dict, label in zip(entry['features'], entry['labels']):
                    # Option 1: p, q, r, ps, qs, rs
                    feat1 = [
                        feat_dict['p'],
                        feat_dict['q'],
                        feat_dict['r'],
                        feat_dict['ps'],
                        feat_dict['qs'],
                        feat_dict['rs']
                    ]
                    features_option1.append(feat1)
                    
                    # Option 2: activation_base + activation_source (flattened)
                    activation_base = np.array(feat_dict['activation_base'])
                    activation_source = np.array(feat_dict['activation_source'])
                    feat2 = np.concatenate([activation_base, activation_source]).tolist()
                    features_option2.append(feat2)
                    
                    labels_list.append(label)
    
    # Convert to numpy arrays
    X_option1 = np.array(features_option1)
    X_option2 = np.array(features_option2)
    y = np.array(labels_list)
    
    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(y)}")
    print(f"  Option 1 features shape: {X_option1.shape} (p, q, r, ps, qs, rs)")
    print(f"  Option 2 features shape: {X_option2.shape} (activation_base + activation_source)")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Select features based on argument
    if args.features == 'pqr':
        print("\n" + "=" * 60)
        print("Using Features: [p, q, r, ps, qs, rs]")
        print("=" * 60)
        
        # Create dataframe
        df = pd.DataFrame(
            X_option1,
            columns=['p', 'q', 'r', 'ps', 'qs', 'rs']
        )
        df['label'] = y
        
    elif args.features == 'activation':
        print("\n" + "=" * 60)
        print("Using Features: [activation_base, activation_source] (flattened)")
        print("=" * 60)
        
        # Create dataframe
        n_base = len(features_option2[0]) // 2
        feature_names = (
            [f'activation_base_{i}' for i in range(n_base)] +
            [f'activation_source_{i}' for i in range(n_base)]
        )
        df = pd.DataFrame(X_option2, columns=feature_names)
        df['label'] = y
    
    # Prepare model arguments
    model_kwargs = {
        'label_col': 'label',
        'model_type': args.method,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'max_iter': args.max_iter
    }
    
    # Add class_weight for both logistic regression and neural networks
    class_weight = None if args.class_weight == 'none' else args.class_weight
    model_kwargs['class_weight'] = class_weight
    
    # Add neural network specific arguments
    if args.method == 'neural_net':
        model_kwargs['hidden_layer_sizes'] = tuple(args.hidden_layers)
    
    # Train model and report test accuracy
    model, scaler, results = learn_hyperplane(df, **model_kwargs)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Feature Set: {args.features}")
    print(f"Method: {args.method}")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy (unweighted): {results['test_accuracy']:.4f}")
    print(f"Number of features: {results['n_features']}")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Class distribution: {results['class_distribution']}")
    print(f"\nTest Set Prediction Distribution:")
    for pred_class in sorted(results['test_prediction_distribution'].keys()):
        dist = results['test_prediction_distribution'][pred_class]
        print(f"  Class {pred_class}: {dist['count']} ({dist['percentage']:.2f}%)")
    print(f"{'='*60}")
