import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import warnings
import json
import os
import argparse
warnings.filterwarnings('ignore')


class BucklingNet(nn.Module):
    """
    Neural Network that learns a mapping h_θ: X → Δ^(K-1),
    where K is the number of desired buckles (buckets).
    """
    def __init__(self, input_dim: int, num_buckets: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        """
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        num_buckets : int
            Number of buckets (K) for partitioning
        hidden_dims : Tuple[int, ...]
            Hidden layer dimensions. Default: (64, 64)
        """
        super(BucklingNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer: maps to num_buckets (will be softmaxed)
        layers.append(nn.Linear(prev_dim, num_buckets))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps input to probability distribution over buckets.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (N, input_dim)
        
        Returns:
        --------
        z : torch.Tensor
            Soft assignments to buckets, shape (N, num_buckets)
        """
        logits = self.net(x)
        z = F.softmax(logits, dim=-1)
        return z


def buckling_loss(
    z_s: torch.Tensor,
    z_b: torch.Tensor,
    y: torch.Tensor,
    lambda_bal: float = 1.0,
    lambda_sharp: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the buckling loss with three components:
    1. Purity Penalty: Minimizes variance of buckle scores around 0.5
    2. Balancing Penalty: Prevents collapse to single buckle
    3. Sharpness Penalty: Forces hard decisions
    
    Parameters:
    -----------
    z_s : torch.Tensor
        Assignments for base features s_i, shape (N, K)
    z_b : torch.Tensor
        Assignments for source features b_i, shape (N, K)
    y : torch.Tensor
        Labels, shape (N,) or (N, 1)
    lambda_bal : float
        Weight for balancing penalty
    lambda_sharp : float
        Weight for sharpness penalty
    
    Returns:
    --------
    total_loss : torch.Tensor
        Total loss value
    q_hat : torch.Tensor
        Estimated scores for each buckle, shape (K,)
    """
    N, K = z_s.shape
    
    # Ensure y is 1D
    if y.dim() > 1:
        y = y.squeeze()
    
    # 1. Calculate Estimated Buckle Scores (q_hat)
    # Probability that both s_i and b_i are in buckle k
    w = z_s * z_b  # (N, K)
    
    numerator = torch.sum(w * y.unsqueeze(1), dim=0)  # (K,)
    denominator = torch.sum(w, dim=0) + 1e-8  # (K,)
    q_hat = numerator / denominator  # (K,)
    
    # 2. Purity Penalty: Push q_hat to 0 or 1
    # L_P = (1/K) * sum_k q_hat_k * (1 - q_hat_k)
    loss_purity = torch.mean(q_hat) #torch.mean(q_hat * (1 - q_hat))
    
    # 3. Balancing Penalty: Encourage uniform distribution across buckets
    # L_B = sum_k z_bar_k * log(z_bar_k)
    # where z_bar_k = (1/(2N)) * sum_i (z_k(s_i) + z_k(b_i))
    mean_z = torch.mean(0.5 * (z_s + z_b), dim=0)  # (K,)
    loss_balance = torch.sum(mean_z * torch.log(mean_z + 1e-8))
    
    # 4. Sharpness Penalty: Force z_i to be one-hot
    # L_S = -(1/(2N)) * sum_i sum_k z_k(x_i) * log(z_k(x_i))
    entropy_s = -torch.sum(z_s * torch.log(z_s + 1e-8), dim=1)  # (N,)
    entropy_b = -torch.sum(z_b * torch.log(z_b + 1e-8), dim=1)  # (N,)
    loss_sharp = torch.mean(0.5 * (entropy_s + entropy_b))
    
    total_loss = loss_purity + lambda_bal * loss_balance + lambda_sharp * loss_sharp
    
    return total_loss, q_hat


class CausalDataset(Dataset):
    """Dataset class for causal abstraction data."""
    def __init__(self, s_features: np.ndarray, b_features: np.ndarray, labels: np.ndarray):
        """
        Parameters:
        -----------
        s_features : np.ndarray
            Base features (p, q, r or activation_base), shape (N, dim_s)
        b_features : np.ndarray
            Source features (ps, qs, rs or activation_source), shape (N, dim_b)
        labels : np.ndarray
            Binary labels, shape (N,)
        """
        self.s_features = torch.FloatTensor(s_features)
        self.b_features = torch.FloatTensor(b_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            's': self.s_features[idx],
            'b': self.b_features[idx],
            'y': self.labels[idx]
        }


def train_buckling_model(
    df: pd.DataFrame,
    label_col: str = 'label',
    num_buckets: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    lambda_bal: float = 1.0,
    lambda_sharp: float = 0.1,
    hidden_dims: Tuple[int, ...] = (64, 64),
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[BucklingNet, StandardScaler, Dict]:
    """
    Train a buckling network to partition the feature space.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and labels
    label_col : str, default='label'
        Name of the column containing binary labels (0 or 1)
    num_buckets : int, default=5
        Number of buckets (K) for partitioning
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    batch_size : int, default=64
        Batch size for training
    num_epochs : int, default=100
        Number of training epochs
    learning_rate : float, default=1e-3
        Learning rate for optimizer
    lambda_bal : float, default=1.0
        Weight for balancing penalty
    lambda_sharp : float, default=0.1
        Weight for sharpness penalty
    hidden_dims : Tuple[int, ...], default=(64, 64)
        Hidden layer dimensions
    normalize : bool, default=True
        Whether to standardize features
    device : torch.device or None
        Device to run training on (None = auto-detect)
    
    Returns:
    --------
    model : BucklingNet
        Trained model
    scaler : StandardScaler or None
        Feature scaler (None if normalize=False)
    results : dict
        Dictionary containing training results and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate input
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")
    
    # Check for binary labels
    unique_labels = df[label_col].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")
    
    # Separate base features (s) and source features (b)
    # For pqr: s = [p, q, r], b = [ps, qs, rs]
    # For activation: s = activation_base, b = activation_source
    feature_cols = [col for col in df.columns if col != label_col]
    
    # Determine feature split based on column names
    if 'p' in feature_cols and 'ps' in feature_cols:
        # pqr features
        s_cols = ['p', 'q', 'r']
        b_cols = ['ps', 'qs', 'rs']
    else:
        # activation features - split in half
        n_features = len(feature_cols)
        n_base = n_features // 2
        s_cols = feature_cols[:n_base]
        b_cols = feature_cols[n_base:]
    
    s_features = df[s_cols].values
    b_features = df[b_cols].values
    labels = df[label_col].values
    
    # Get dataset info
    n_samples = len(labels)
    class_counts = pd.Series(labels).value_counts().to_dict()
    
    print(f"Dataset info:")
    print(f"  Samples: {n_samples}")
    print(f"  Base features (s) shape: {s_features.shape}")
    print(f"  Source features (b) shape: {b_features.shape}")
    print(f"  Class distribution: {class_counts}")
    
    # Split data
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    s_train, s_test = s_features[train_idx], s_features[test_idx]
    b_train, b_test = b_features[train_idx], b_features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Normalize features if requested
    scaler_s = None
    scaler_b = None
    if normalize:
        scaler_s = StandardScaler()
        scaler_b = StandardScaler()
        s_train = scaler_s.fit_transform(s_train)
        s_test = scaler_s.transform(s_test)
        b_train = scaler_b.fit_transform(b_train)
        b_test = scaler_b.transform(b_test)
    
    # Create datasets and dataloaders
    train_dataset = CausalDataset(s_train, b_train, y_train)
    test_dataset = CausalDataset(s_test, b_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = s_features.shape[1]  # Assume s and b have same dimension
    model = BucklingNet(input_dim, num_buckets, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining buckling network:")
    print(f"  Device: {device}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of buckets: {num_buckets}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda balance: {lambda_bal}")
    print(f"  Lambda sharp: {lambda_sharp}")
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            s_batch = batch['s'].to(device)
            b_batch = batch['b'].to(device)
            y_batch = batch['y'].to(device)
            
            # Forward pass
            z_s = model(s_batch)
            z_b = model(b_batch)
            
            # Compute loss
            loss, _ = buckling_loss(z_s, z_b, y_batch, lambda_bal, lambda_sharp)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        all_q_hat = []
        with torch.no_grad():
            for batch in test_loader:
                s_batch = batch['s'].to(device)
                b_batch = batch['b'].to(device)
                y_batch = batch['y'].to(device)
                
                z_s = model(s_batch)
                z_b = model(b_batch)
                
                loss, q_hat = buckling_loss(z_s, z_b, y_batch, lambda_bal, lambda_sharp)
                test_loss += loss.item()
                all_q_hat.append(q_hat.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Compute average q_hat across all test batches
        avg_q_hat = np.mean(np.stack(all_q_hat), axis=0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Q_hat: {avg_q_hat}")
    
    # Final evaluation
    model.eval()
    final_q_hat = []
    all_z_s = []
    all_z_b = []
    with torch.no_grad():
        for batch in test_loader:
            s_batch = batch['s'].to(device)
            b_batch = batch['b'].to(device)
            y_batch = batch['y'].to(device)
            
            z_s = model(s_batch)
            z_b = model(b_batch)
            
            _, q_hat = buckling_loss(z_s, z_b, y_batch, lambda_bal, lambda_sharp)
            final_q_hat.append(q_hat.cpu().numpy())
            all_z_s.append(z_s.cpu().numpy())
            all_z_b.append(z_b.cpu().numpy())
    
    final_q_hat = np.mean(np.stack(final_q_hat), axis=0)
    
    # Compute bucket proportions
    # Concatenate all batches
    z_s_all = np.concatenate(all_z_s, axis=0)  # (N_test, K)
    z_b_all = np.concatenate(all_z_b, axis=0)  # (N_test, K)
    
    # Proportion for each bucket: average assignment probability
    bucket_prop_s = np.mean(z_s_all, axis=0)  # (K,)
    bucket_prop_b = np.mean(z_b_all, axis=0)  # (K,)
    bucket_prop_combined = np.mean(0.5 * (z_s_all + z_b_all), axis=0)  # (K,)
    
    print(f"\nTraining completed!")
    print(f"  Final test loss: {test_losses[-1]:.4f}")
    print(f"  Final bucket scores (q_hat): {final_q_hat}")
    print(f"  Bucket proportions (base features s): {bucket_prop_s}")
    print(f"  Bucket proportions (source features b): {bucket_prop_b}")
    print(f"  Bucket proportions (combined): {bucket_prop_combined}")
    
    # Prepare results dictionary
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_test_loss': test_losses[-1],
        'final_q_hat': final_q_hat.tolist(),
        'bucket_proportions_s': bucket_prop_s.tolist(),
        'bucket_proportions_b': bucket_prop_b.tolist(),
        'bucket_proportions_combined': bucket_prop_combined.tolist(),
        'num_buckets': num_buckets,
        'n_features': input_dim,
        'n_samples': n_samples,
        'class_distribution': class_counts,
        'feature_names': feature_cols,
        's_cols': s_cols,
        'b_cols': b_cols
    }
    
    return model, (scaler_s, scaler_b), results


def predict_buckets(
    model: BucklingNet,
    s_features: np.ndarray,
    b_features: np.ndarray,
    scaler_s: Optional[StandardScaler] = None,
    scaler_b: Optional[StandardScaler] = None,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict bucket assignments for new data.
    
    Parameters:
    -----------
    model : BucklingNet
        Trained model
    s_features : np.ndarray
        Base features, shape (N, dim_s)
    b_features : np.ndarray
        Source features, shape (N, dim_b)
    scaler_s : StandardScaler or None
        Scaler for base features
    scaler_b : StandardScaler or None
        Scaler for source features
    device : torch.device or None
        Device to run inference on
    
    Returns:
    --------
    z_s : np.ndarray
        Bucket assignments for s, shape (N, K)
    z_b : np.ndarray
        Bucket assignments for b, shape (N, K)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize if needed
    if scaler_s is not None:
        s_features = scaler_s.transform(s_features)
    if scaler_b is not None:
        b_features = scaler_b.transform(b_features)
    
    # Convert to tensors
    s_tensor = torch.FloatTensor(s_features).to(device)
    b_tensor = torch.FloatTensor(b_features).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        z_s = model(s_tensor).cpu().numpy()
        z_b = model(b_tensor).cpu().numpy()
    
    return z_s, z_b


def analyze_pqr_combinations_by_bucket(
    model: BucklingNet,
    df: pd.DataFrame,
    scaler_s: Optional[StandardScaler] = None,
    scaler_b: Optional[StandardScaler] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Analyze which pqr combinations end up in each bucket.
    
    Parameters:
    -----------
    model : BucklingNet
        Trained model
    df : pd.DataFrame
        Dataframe containing p, q, r, ps, qs, rs columns
    scaler_s : StandardScaler or None
        Scaler for base features (p, q, r)
    scaler_b : StandardScaler or None
        Scaler for source features (ps, qs, rs)
    device : torch.device or None
        Device to run inference on
    
    Returns:
    --------
    analysis : dict
        Dictionary containing bucket assignments and pqr combinations
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract pqr features (original unscaled values for combination tracking)
    s_features_original = df[['p', 'q', 'r']].values
    b_features_original = df[['ps', 'qs', 'rs']].values
    
    # Get bucket assignments (this will scale internally if scalers provided)
    z_s, z_b = predict_buckets(model, s_features_original, b_features_original, scaler_s, scaler_b, device)
    
    # Get hard assignments (argmax)
    bucket_assignments_s = np.argmax(z_s, axis=1)  # (N,)
    bucket_assignments_b = np.argmax(z_b, axis=1)  # (N,)
    
    # Get pqr combinations from original unscaled values (convert to tuples for hashing)
    pqr_combinations = [tuple(row) for row in s_features_original.astype(int)]
    
    # Group by bucket
    num_buckets = z_s.shape[1]
    bucket_to_pqr_s = {i: [] for i in range(num_buckets)}
    bucket_to_pqr_b = {i: [] for i in range(num_buckets)}
    
    for idx, (bucket_s, bucket_b, pqr) in enumerate(zip(bucket_assignments_s, bucket_assignments_b, pqr_combinations)):
        bucket_to_pqr_s[bucket_s].append(pqr)
        bucket_to_pqr_b[bucket_b].append(pqr)
    
    # Count unique combinations per bucket
    bucket_to_unique_pqr_s = {i: set(pqr_list) for i, pqr_list in bucket_to_pqr_s.items()}
    bucket_to_unique_pqr_b = {i: set(pqr_list) for i, pqr_list in bucket_to_pqr_b.items()}
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PQR COMBINATIONS BY BUCKET (Base Features s: p, q, r)")
    print(f"{'='*60}")
    for bucket_id in range(num_buckets):
        unique_combos = sorted(bucket_to_unique_pqr_s[bucket_id])
        count = len(bucket_to_pqr_s[bucket_id])
        print(f"\nBucket {bucket_id} ({count} samples):")
        if unique_combos:
            for combo in unique_combos:
                combo_count = bucket_to_pqr_s[bucket_id].count(combo)
                print(f"  {combo}: {combo_count} samples")
        else:
            print(f"  (empty)")
    
    print(f"\n{'='*60}")
    
    return {
        'bucket_to_pqr_s': bucket_to_pqr_s,
        'bucket_to_pqr_b': bucket_to_pqr_b,
        'bucket_to_unique_pqr_s': {k: list(v) for k, v in bucket_to_unique_pqr_s.items()},
        'bucket_to_unique_pqr_b': {k: list(v) for k, v in bucket_to_unique_pqr_b.items()},
        'bucket_assignments_s': bucket_assignments_s.tolist(),
        'bucket_assignments_b': bucket_assignments_b.tolist()
    }


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Train buckling network for causal abstraction dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--features', '-f',
        type=str,
        choices=['pqr', 'activation'],
        default='pqr',
        help='Feature set: pqr (p,q,r,ps,qs,rs) or activation (activation_base + activation_source)'
    )
    parser.add_argument(
        '--num-buckets', '-k',
        type=int,
        default=5,
        help='Number of buckets (K) for partitioning'
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
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--lambda-bal',
        type=float,
        default=1.0,
        help='Weight for balancing penalty'
    )
    parser.add_argument(
        '--lambda-sharp',
        type=float,
        default=0.1,
        help='Weight for sharpness penalty'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[64, 64],
        help='Hidden layer dimensions (e.g., --hidden-dims 64 64)'
    )
    
    args = parser.parse_args()
    
    # Load dataset from JSON file
    print("=" * 60)
    print("Input Classifier - Buckling Network")
    print("=" * 60)
    print(f"Features: {args.features}")
    print(f"Number of buckets: {args.num_buckets}")
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
        'num_buckets': args.num_buckets,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'lambda_bal': args.lambda_bal,
        'lambda_sharp': args.lambda_sharp,
        'hidden_dims': tuple(args.hidden_dims)
    }
    
    # Train model
    model, scalers, results = train_buckling_model(df, **model_kwargs)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Feature Set: {args.features}")
    print(f"Number of Buckets: {args.num_buckets}")
    print(f"Final Test Loss: {results['final_test_loss']:.4f}")
    print(f"Final Bucket Scores (q_hat): {results['final_q_hat']}")
    print(f"Bucket Proportions (base features s): {results['bucket_proportions_s']}")
    print(f"Bucket Proportions (combined): {results['bucket_proportions_combined']}")
    print(f"Number of features: {results['n_features']}")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Class distribution: {results['class_distribution']}")
    print(f"{'='*60}")
    
    # Analyze pqr combinations by bucket if using pqr features
    if args.features == 'pqr':
        scaler_s, scaler_b = scalers
        pqr_analysis = analyze_pqr_combinations_by_bucket(
            model, df, scaler_s, scaler_b
        )
