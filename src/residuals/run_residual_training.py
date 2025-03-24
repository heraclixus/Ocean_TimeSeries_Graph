import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from residual_dataset import SSTResidualDatasetLoader

class MLP(nn.Module):
    def __init__(self, seq_len=12, feature_dim=20, hidden_dims=[256, 128], dropout_rate=0.2):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        input_dim = seq_len * feature_dim
        
        # Create a list of layers with dropout
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout after activation
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, seq_len * feature_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size = x.shape[0]
        # Flatten the sequence and feature dimensions
        x_flat = x.reshape(batch_size, -1)
        # Apply MLP
        output = self.model(x_flat)
        # Reshape back to (batch_size, seq_len, feature_dim)
        return output.reshape(batch_size, self.seq_len, self.feature_dim)


class TransformerModel(nn.Module):
    def __init__(self, feature_dim=20, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.feature_dim = feature_dim
        
        # Input embedding layer
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)  # (batch_size, seq_len, feature_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss
    Weights each dimension of the output by its standard deviation in the training data
    with additional emphasis on specified important dimensions
    """
    def __init__(self, weights, important_dims=None, importance_factor=2.0):
        super(WeightedMSELoss, self).__init__()
        # Ensure weights has the right shape for broadcasting
        self.weights = weights.clone()
        
        # Apply additional weighting to important dimensions
        if important_dims is not None:
            for dim in important_dims:
                if dim < len(self.weights):
                    self.weights[dim] *= importance_factor
            
            # Re-normalize weights to maintain overall scale
            self.weights = self.weights * (len(weights) / self.weights.sum())
    
    def forward(self, outputs, targets):
        # outputs and targets: (batch_size, seq_len, feature_dim)
        # Calculate squared error
        squared_error = (outputs - targets) ** 2
        
        # Average over batch and sequence dimensions, keeping feature dimension
        # Shape becomes (feature_dim,)
        mse_per_dim = torch.mean(squared_error, dim=(0, 1))
        
        # Apply weights to each dimension
        weighted_mse = mse_per_dim * self.weights
        
        # Return mean of weighted errors
        return torch.mean(weighted_mse)


def compute_dimension_weights(X, important_dims=None, importance_factor=2.0):
    """
    Compute weights for each dimension based on standard deviation
    Higher standard deviation -> higher weight
    
    Args:
        X: Training data of shape (samples, seq_len, feature_dim)
        important_dims: List of dimension indices to apply additional weighting to
        importance_factor: Factor to multiply important dimension weights by
        
    Returns:
        weights: Tensor of shape (feature_dim,)
    """
    # Reshape to (samples*seq_len, feature_dim)
    X_flat = X.reshape(-1, X.shape[-1])
    
    # Compute std dev for each dimension
    std_per_dim = torch.std(X_flat, dim=0)
    
    # Normalize weights to sum to feature_dim (to keep overall loss magnitude similar)
    feature_dim = X.shape[-1]
    normalized_weights = (std_per_dim / torch.sum(std_per_dim)) * feature_dim
    
    return normalized_weights


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_name, 
                scheduler=None, early_stopping_patience=10, l1_lambda=0.0, args=None, save_figs=True):
    train_losses = []
    val_losses = []
    per_dim_train_rmse = []
    per_dim_val_rmse = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        per_dim_error_sum = np.zeros(model.feature_dim)
        per_dim_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
            
            # Calculate per-dimension RMSE
            with torch.no_grad():
                error = (outputs - y_batch).detach().cpu().numpy() ** 2
                per_dim_error_sum += np.sqrt(np.mean(error, axis=(0, 1)))
                per_dim_count += 1
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * X_batch.size(0)
            pbar.set_postfix({"loss": loss.item()})
        
        # Compute average loss and per-dimension RMSE
        train_loss /= len(train_loader.dataset)
        per_dim_train_rmse.append(per_dim_error_sum / per_dim_count)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        per_dim_error_sum = np.zeros(model.feature_dim)
        per_dim_count = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Calculate per-dimension RMSE
                error = (outputs - y_batch).cpu().numpy() ** 2
                per_dim_error_sum += np.sqrt(np.mean(error, axis=(0, 1)))
                per_dim_count += 1
                
                # Update metrics
                val_loss += loss.item() * X_batch.size(0)
                pbar.set_postfix({"loss": loss.item()})
        
        # Compute average validation loss and per-dimension RMSE
        val_loss /= len(val_loader.dataset)
        per_dim_val_rmse.append(per_dim_error_sum / per_dim_count)
        val_losses.append(val_loss)
        
        # Update the learning rate scheduler with validation loss
        if scheduler:
            scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate a filename with hyperparameters
    if args is not None and args.model == 'transformer':
        hyperparam_str = f"h{args.hidden_dim}_head{args.num_heads}_l{args.num_layers}_d{args.dropout}_lr{args.lr}_cf{args.critical_importance_factor}"
    else:
        hyperparam_str = "default"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"results/{model_name}_{hyperparam_str}_{timestamp}.pt"
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'val_loss': best_val_loss,
        'hyperparams': vars(args) if args is not None else {},
        'epochs_trained': len(train_losses),
    }, model_filename)
    
    print(f"Model saved to {model_filename}")
    
    # Only save figures if specifically requested (e.g., for best model in grid search)
    if save_figs:
        # Create and save plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Weighted MSE)')
        plt.title(f'{model_name} Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.array(per_dim_train_rmse).mean(axis=1), label='Training RMSE')
        plt.plot(np.array(per_dim_val_rmse).mean(axis=1), label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (average across dimensions)')
        plt.title(f'{model_name} RMSE Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_loss_curve_{hyperparam_str}_{timestamp}.png")
        
        # Plot per-dimension RMSE
        plt.figure(figsize=(14, 8))
        final_train_rmse = per_dim_train_rmse[-1]
        final_val_rmse = per_dim_val_rmse[-1]
        
        x = np.arange(model.feature_dim)
        width = 0.35
        
        plt.bar(x - width/2, final_train_rmse, width, label='Training RMSE')
        plt.bar(x + width/2, final_val_rmse, width, label='Validation RMSE')
        
        plt.xlabel('Dimension')
        plt.ylabel('RMSE')
        plt.title(f'{model_name} Per-Dimension RMSE')
        plt.xticks(x)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_per_dim_rmse_{hyperparam_str}_{timestamp}.png")
        
        # Plot per-dimension RMSE over epochs
        plt.figure(figsize=(15, 10))
        
        for dim in range(model.feature_dim):
            plt.subplot(5, 4, dim+1)
            plt.plot([rmse[dim] for rmse in per_dim_train_rmse], label='Train')
            plt.plot([rmse[dim] for rmse in per_dim_val_rmse], label='Val')
            plt.title(f'Dim {dim}')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            if dim == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_per_dim_rmse_epochs_{hyperparam_str}_{timestamp}.png")
        
        print(f"Saved figures for model {hyperparam_str}")
    else:
        print("Skipping figure generation as requested")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'per_dim_train_rmse': per_dim_train_rmse,
        'per_dim_val_rmse': per_dim_val_rmse,
        'best_val_loss': best_val_loss,
        'model_filename': model_filename,
        'hyperparam_str': hyperparam_str,
        'timestamp': timestamp
    }


def add_noise_to_data(data, noise_level=0.05):
    """
    Add Gaussian noise to data for augmentation
    
    Args:
        data: Input data tensor
        noise_level: Standard deviation of the noise as a fraction of data std
    
    Returns:
        Noisy data tensor of the same shape
    """
    data_std = torch.std(data)
    noise = torch.randn_like(data) * data_std * noise_level
    return data + noise


# Add normalization class
class DataNormalizer:
    """
    Normalizes data across specified dimensions and keeps track of normalization parameters
    to allow for denormalization.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.initialized = False
    
    def fit(self, data):
        """
        Compute mean and standard deviation from data
        
        Args:
            data: Tensor of shape (*, feature_dim) where * can be any number of dimensions
        """
        # Reshape to (-1, feature_dim) to compute stats over all but the last dimension
        reshaped_data = data.reshape(-1, data.shape[-1])
        self.mean = torch.mean(reshaped_data, dim=0)
        self.std = torch.std(reshaped_data, dim=0)
        # Prevent division by zero
        self.std[self.std < 1e-5] = 1.0
        self.initialized = True
        return self
    
    def normalize(self, data):
        """
        Normalize data using stored mean and std
        
        Args:
            data: Tensor of any shape, but last dimension must match feature_dim
        
        Returns:
            Normalized data with same shape
        """
        if not self.initialized:
            raise ValueError("Normalizer must be initialized with fit() before normalizing data")
        
        # Reshape to keep original dimensions but normalize along feature dimension
        original_shape = data.shape
        reshaped_data = data.reshape(-1, original_shape[-1])
        normalized_data = (reshaped_data - self.mean) / self.std
        return normalized_data.reshape(original_shape)
    
    def denormalize(self, normalized_data):
        """
        Denormalize data using stored mean and std
        
        Args:
            normalized_data: Normalized tensor of any shape
        
        Returns:
            Original-scale data with same shape
        """
        if not self.initialized:
            raise ValueError("Normalizer must be initialized with fit() before denormalizing data")
        
        # Reshape to keep original dimensions but denormalize along feature dimension
        original_shape = normalized_data.shape
        reshaped_data = normalized_data.reshape(-1, original_shape[-1])
        denormalized_data = (reshaped_data * self.std) + self.mean
        return denormalized_data.reshape(original_shape)


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine whether to save figures:
    # - If force_save_figs is true, always save
    # - If hyperparameter_search is true and force_save_figs is false, don't save
    # - Otherwise use save_figs flag, defaulting to True if not specified
    save_figs = args.force_save_figs or (not args.hyperparameter_search and args.save_figs)
    
    # Load the residual dataset
    data_loader = SSTResidualDatasetLoader(
        data_file_path=args.data_file,
        residuals_file_path=args.residuals_file,
        n_pcs=args.n_pcs,
        train_length=args.train_length
    )
    
    # Get data
    X_train = data_loader.X_train  # (train_length, 12, 20)
    y_train = data_loader.y_train  # (train_length, 12, 20)
    X_test = data_loader.X_test    # (test_length, 12, 20)
    y_test = data_loader.y_test    # (test_length, 12, 20)
    
    # Split training data into train and validation sets
    val_size = int(len(X_train) * args.val_split)
    train_size = len(X_train) - val_size
    
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Initialize normalizers (even if not used, to avoid conditional logic later)
    X_normalizer = DataNormalizer()
    y_normalizer = DataNormalizer()
    
    # Apply normalization if enabled
    if args.normalize:
        print("\nNormalizing data across feature dimensions...")
        # Normalize inputs
        X_normalizer.fit(X_train_tensor)
        X_train_tensor = X_normalizer.normalize(X_train_tensor)
        X_val_tensor = X_normalizer.normalize(X_val_tensor)
        X_test_tensor = X_normalizer.normalize(X_test_tensor)
        
        # Normalize targets (optional, but helps with loss calculation)
        y_normalizer.fit(y_train_tensor)
        y_train_tensor = y_normalizer.normalize(y_train_tensor)
        y_val_tensor = y_normalizer.normalize(y_val_tensor)
        y_test_tensor = y_normalizer.normalize(y_test_tensor)
        
        print("Data normalization complete.")
        print("Input data stats:")
        print(f"  Mean: min={X_normalizer.mean.min().item():.4f}, max={X_normalizer.mean.max().item():.4f}")
        print(f"  Std: min={X_normalizer.std.min().item():.4f}, max={X_normalizer.std.max().item():.4f}")
    else:
        # If normalization is not used, we still initialize the normalizers
        # with dummy fits so that denormalize() becomes a no-op
        X_normalizer.fit(torch.zeros_like(X_train_tensor[0:1]))
        y_normalizer.fit(torch.zeros_like(y_train_tensor[0:1]))
        
        # Modify the denormalize method to return the input unchanged when not initialized properly
        X_normalizer.initialized = False
        y_normalizer.initialized = False
    
    # Data augmentation if enabled
    if args.data_augmentation:
        print("\nPerforming data augmentation...")
        augmented_X = []
        augmented_y = []
        
        # Create multiple augmented versions of the training data
        for _ in range(args.augmentation_factor):
            noisy_X = add_noise_to_data(X_train_tensor, args.noise_level)
            augmented_X.append(noisy_X)
            augmented_y.append(y_train_tensor.clone())
        
        # Combine original and augmented data
        augmented_X.append(X_train_tensor)
        augmented_y.append(y_train_tensor)
        
        X_train_tensor = torch.cat(augmented_X, dim=0)
        y_train_tensor = torch.cat(augmented_y, dim=0)
        
        print(f"Augmented train set size: {len(X_train_tensor)}")
    
    # Compute dimension weights based on standard deviation
    # Note: If using normalized data, all std devs will be ~1, so importance factor becomes more important
    # For weighting, we should use the original scale data to get proper variance-based weights
    if args.normalize:
        # Use denormalized data for computing weights
        weight_data = y_normalizer.denormalize(y_train_tensor)
    else:
        weight_data = y_train_tensor
        
    dim_weights = compute_dimension_weights(weight_data)
    
    # Set importance factors for different dimensions
    # First 5 dimensions get the base importance factor
    important_dims = list(range(5))
    importance_factor = args.importance_factor
    
    # Dimensions 0 and 1 get an extra boost with higher importance factor
    critical_dims = [0, 1]
    critical_importance_factor = args.critical_importance_factor
    
    # Apply extra weighting to important dimensions
    enhanced_weights = dim_weights.clone()
    
    # First apply the regular importance factor to all high-priority dimensions
    for dim in important_dims:
        if dim < len(enhanced_weights):
            enhanced_weights[dim] *= importance_factor
    
    # Then apply the critical importance factor to dimensions 0 and 1
    for dim in critical_dims:
        if dim < len(enhanced_weights):
            # Reset to original weight first, then apply critical factor
            # This way we don't compound the factors
            enhanced_weights[dim] = dim_weights[dim] * critical_importance_factor
    
    # Re-normalize to maintain overall scale
    enhanced_weights = enhanced_weights * (len(dim_weights) / enhanced_weights.sum())
    
    print("\nDimension weights based on standard deviation:")
    print(f"Dimensions 0-4 have been given {importance_factor:.1f}x importance")
    print(f"Dimensions 0-1 have been given an additional boost to {critical_importance_factor:.1f}x importance")
    
    for dim, weight in enumerate(enhanced_weights.numpy()):
        if dim in critical_dims:
            marker = "** "  # Double asterisk for critical dimensions
        elif dim in important_dims:
            marker = "* "   # Single asterisk for important dimensions
        else:
            marker = "  "   # No asterisk for regular dimensions
        print(f"{marker}Dim {dim}: {weight:.4f}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model based on argument
    if args.model == 'mlp':
        model = MLP(
            seq_len=12,
            feature_dim=args.n_pcs,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout
        ).to(device)
        model_name = 'MLP'
    elif args.model == 'transformer':
        model = TransformerModel(
            feature_dim=args.n_pcs,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        model_name = 'Transformer'
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print(f"Model: {model_name}")
    print(model)
    
    # Loss function with dimension weighting and importance on first 5 dimensions
    criterion = WeightedMSELoss(enhanced_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Train the model
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_name=model_name,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping,
        l1_lambda=args.l1_lambda,
        args=args,  # Pass args to include in file names
        save_figs=save_figs
    )
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    per_dim_error_sum = np.zeros(args.n_pcs)
    unweighted_mse = nn.MSELoss()  # For comparing standard MSE
    unweighted_test_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # Calculate weighted loss (in normalized space)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            
            # Calculate unweighted loss (in normalized space)
            std_loss = unweighted_mse(outputs, y_batch)
            unweighted_test_loss += std_loss.item() * X_batch.size(0)
            
            # For RMSE calculation, denormalize outputs and targets if normalization was used
            if args.normalize:
                # Move data back to CPU for denormalization
                outputs_cpu = outputs.cpu()
                y_batch_cpu = y_batch.cpu()
                
                # Denormalize to original scale
                outputs_denorm = y_normalizer.denormalize(outputs_cpu)
                y_batch_denorm = y_normalizer.denormalize(y_batch_cpu)
                
                # Calculate per-dimension RMSE in original scale
                error = (outputs_denorm - y_batch_denorm).numpy() ** 2
            else:
                # If not normalized, calculate RMSE directly
                error = (outputs.cpu() - y_batch.cpu()).numpy() ** 2
                
            per_dim_error_sum += np.sqrt(np.mean(error, axis=(0, 1)))
    
    # Compute average test loss and per-dimension RMSE
    test_loss /= len(test_loader.dataset)
    unweighted_test_loss /= len(test_loader.dataset)
    per_dim_test_rmse = per_dim_error_sum / len(test_loader)
    
    print(f"\nTest Results for {model_name}:")
    print(f"Test Loss (Weighted MSE): {test_loss:.6f}")
    print(f"Test Loss (Standard MSE): {unweighted_test_loss:.6f}")
    
    # If normalized, note that the loss values are in normalized space
    if args.normalize:
        print("Note: Loss values are in normalized space")
        
    print(f"Test RMSE (overall): {np.mean(per_dim_test_rmse):.6f}")
    print(f"Per-dimension RMSE (test): {'Original scale' if args.normalize else 'Same scale as training'}")
    
    # Create a DataFrame to display dimension statistics side by side
    # Add a new Priority category for critical dimensions
    priority_labels = ['Critical' if i in critical_dims else 
                      ('High' if i in important_dims else 'Normal') 
                      for i in range(args.n_pcs)]
    
    dim_stats = pd.DataFrame({
        'Dimension': range(args.n_pcs),
        'Priority': priority_labels,
        'Weight': enhanced_weights.cpu().numpy(),
        'RMSE': per_dim_test_rmse,
        'Weighted_RMSE': per_dim_test_rmse * enhanced_weights.cpu().numpy()
    })
    
    print(dim_stats.to_string(index=False))
    
    # Create a bar plot of per-dimension test RMSE with weights overlay
    plt.figure(figsize=(14, 8))
    
    ax1 = plt.gca()
    x = np.arange(args.n_pcs)
    
    # Plot RMSE bars with different colors for different priority dimensions
    colors = ['darkred' if i in critical_dims else 
             ('firebrick' if i in important_dims else 'royalblue') 
             for i in range(args.n_pcs)]
    
    bars = ax1.bar(x, per_dim_test_rmse, width=0.5, color=colors)
    
    # Add a legend patch for dimension priorities
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label='Critical Priority (Dimensions 0-1)'),
        Patch(facecolor='firebrick', label='High Priority (Dimensions 2-4)'),
        Patch(facecolor='royalblue', label='Normal Priority')
    ]
    
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('RMSE' + (' (Original Scale)' if args.normalize else ''))
    ax1.set_title(f'{model_name} Per-Dimension Test RMSE and Weights')
    
    # Create a second y-axis for weights
    ax2 = ax1.twinx()
    ax2.plot(x, enhanced_weights.cpu().numpy(), 'g-', marker='o', linewidth=2, label='Dimension Weight')
    ax2.set_ylabel('Weight')
    
    # Add legends
    legend1 = ax1.legend(handles=legend_elements, loc='upper left')
    ax1.add_artist(legend1)
    ax2.legend(loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/{model_name}_test_rmse_weights_{timestamp}.png")

    # Return the results for hyperparameter search to use
    return train_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train residual models for SST data")
    
    # Data parameters
    parser.add_argument('--data_file', type=str, default="../../data/fcst.mat", help="Path to data file")
    parser.add_argument('--residuals_file', type=str, default="../../data/residual.mat", help="Path to residuals file")
    parser.add_argument('--train_length', type=int, default=700, help="Number of training samples")
    parser.add_argument('--n_pcs', type=int, default=20, help="Number of principal components")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split ratio (increased from 0.1)")
    
    # Add normalization flag
    parser.add_argument('--normalize', action='store_true', help="Normalize data across feature dimensions")
    
    # Model parameters
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer'], help="Model type")
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64], help="Hidden dimensions for MLP (reduced size)")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension for Transformer (reduced size)")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of transformer layers")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate (increased from 0.1)")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Maximum number of epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for L2 regularization (increased)")
    parser.add_argument('--l1_lambda', type=float, default=1e-5, help="L1 regularization coefficient")
    parser.add_argument('--early_stopping', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    # Data augmentation
    parser.add_argument('--data_augmentation', action='store_true', help="Enable data augmentation")
    parser.add_argument('--augmentation_factor', type=int, default=2, help="How many augmented copies to create")
    parser.add_argument('--noise_level', type=float, default=0.05, help="Noise level for data augmentation")
    
    # Dimension importance
    parser.add_argument('--importance_factor', type=float, default=2.0, help="Factor to increase weight of important dimensions (dims 2-4)")
    parser.add_argument('--critical_importance_factor', type=float, default=5.0, help="Factor to increase weight of critical dimensions (dims 0-1)")
    
    # Figure saving and hyperparameter search flags
    parser.add_argument('--hyperparameter_search', action='store_true', help="Indicate this is part of a hyperparameter search (disables figure saving)")
    parser.add_argument('--force_save_figs', action='store_true', help="Force saving figures even during hyperparameter search")
    parser.add_argument('--save_figs', action='store_true', help="Save figures (default True unless in hyperparameter search)")
    
    args = parser.parse_args()
    main(args)
