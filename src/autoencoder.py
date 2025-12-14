"""
Autoencoder implementation using PyTorch for dimensionality reduction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple

from .config import (
    RANDOM_SEED,
    AE_ENCODER_LAYERS,
    AE_LATENT_DIM,
    AE_DECODER_LAYERS,
    AE_ACTIVATION,
    AE_OPTIMIZER_LR,
    AE_EPOCHS,
    AE_BATCH_SIZE,
    AE_VALIDATION_SPLIT,
    AE_EARLY_STOPPING_PATIENCE
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AutoencoderModel(nn.Module):
    """
    PyTorch Autoencoder model for dimensionality reduction.
    
    Architecture (from methodology):
    Encoder: Input → 64 (ReLU) → 32 (ReLU) → 10 (Latent, ReLU)
    Decoder: 10 → 32 (ReLU) → 64 (ReLU) → Output (Linear)
    """
    
    def __init__(self, 
                 input_dim: int,
                 encoder_layers: list = AE_ENCODER_LAYERS,
                 latent_dim: int = AE_LATENT_DIM,
                 decoder_layers: list = AE_DECODER_LAYERS):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Number of input features
            encoder_layers: List of hidden layer sizes for encoder [64, 32]
            latent_dim: Size of latent representation (default 10)
            decoder_layers: List of hidden layer sizes for decoder [32, 64]
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_modules = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, hidden_dim))
            encoder_modules.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_modules.append(nn.Linear(prev_dim, latent_dim))
        encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Build decoder
        decoder_modules = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, hidden_dim))
            decoder_modules.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        # Linear activation for output
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Returns:
            (reconstruction, latent): Tuple of reconstructed input and latent representation
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class Autoencoder:
    """
    Wrapper class for autoencoder training and inference.
    Provides a scikit-learn-like interface.
    """
    
    def __init__(self,
                 latent_dim: int = AE_LATENT_DIM,
                 encoder_layers: list = None,
                 decoder_layers: list = None,
                 learning_rate: float = AE_OPTIMIZER_LR,
                 epochs: int = AE_EPOCHS,
                 batch_size: int = AE_BATCH_SIZE,
                 validation_split: float = AE_VALIDATION_SPLIT,
                 early_stopping_patience: int = AE_EARLY_STOPPING_PATIENCE,
                 random_state: int = RANDOM_SEED,
                 verbose: bool = False):
        """
        Initialize the autoencoder wrapper.
        
        Args:
            latent_dim: Size of latent representation
            encoder_layers: Hidden layer sizes for encoder
            decoder_layers: Hidden layer sizes for decoder
            learning_rate: Learning rate for Adam optimizer
            epochs: Maximum training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before early stopping
            random_state: Random seed
            verbose: Whether to print training progress
        """
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers or AE_ENCODER_LAYERS
        self.decoder_layers = decoder_layers or AE_DECODER_LAYERS
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose
        
        self.model: Optional[AutoencoderModel] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_fitted = False
        self.training_history = []
        
    def fit(self, X: np.ndarray) -> 'Autoencoder':
        """
        Train the autoencoder on input data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        set_seed(self.random_state)
        
        input_dim = X.shape[1]
        
        # Initialize model
        self.model = AutoencoderModel(
            input_dim=input_dim,
            encoder_layers=self.encoder_layers,
            latent_dim=self.latent_dim,
            decoder_layers=self.decoder_layers
        ).to(self.device)
        
        # Split data for validation
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        
        # Create data loaders
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)
        
        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                reconstruction, _ = self.model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_x)
            
            train_loss /= len(X_train)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    reconstruction, _ = self.model(batch_x)
                    loss = criterion(reconstruction, batch_x)
                    val_loss += loss.item() * len(batch_x)
            
            val_loss /= len(X_val)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input to latent representation.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Latent representations of shape (n_samples, latent_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before transform")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, latent = self.model(X_tensor)
            return latent.cpu().numpy()
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Input data
            
        Returns:
            Latent representations
        """
        self.fit(X)
        return self.transform(X)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input from its latent representation.
        
        Args:
            X: Input data
            
        Returns:
            Reconstructed data
        """
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before reconstruction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstruction, _ = self.model(X_tensor)
            return reconstruction.cpu().numpy()
    
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate mean reconstruction error (MSE).
        
        Args:
            X: Input data
            
        Returns:
            Mean squared reconstruction error
        """
        X_reconstructed = self.reconstruct(X)
        return np.mean((X - X_reconstructed) ** 2)


if __name__ == "__main__":
    # Test autoencoder
    from .data_loader import load_dataset
    from .preprocessing import preprocess_fold, get_cv_splits
    
    # Load dataset
    X, y, feature_names = load_dataset('CM1')
    print(f"Loaded CM1: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test with first fold
    for fold_idx, (train_idx, test_idx) in enumerate(get_cv_splits(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocess
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True
        )
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Input shape: {X_train_proc.shape}")
        
        # Create and fit autoencoder
        ae = Autoencoder(latent_dim=10, verbose=True)
        X_train_encoded = ae.fit_transform(X_train_proc)
        X_test_encoded = ae.transform(X_test_proc)
        
        print(f"  Encoded train shape: {X_train_encoded.shape}")
        print(f"  Encoded test shape: {X_test_encoded.shape}")
        print(f"  Reconstruction error: {ae.get_reconstruction_error(X_test_proc):.6f}")
        
        # Only test first fold
        break
    
    print("\nAutoencoder test completed successfully!")

