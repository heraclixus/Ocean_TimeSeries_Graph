import torch
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from utils_pca import reconstruct_enso
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.fft as fft

# Import the new dataset class
from pygtemporal_models.graph_dataset_enso import (
    OceanGraphDataset,
    inverse_normalize
)

# Import models
from baseline_models.node import TimeSeriesNODE, NeuralODEForecaster
from baseline_models.graphode import GraphNeuralODE, NeuralGDEForecaster
from baseline_models.utils import save_results
from pygtemporal_models.pyg_temp_dataset import batch_data_to_timeseries

# Import additional models from pygtemporal_models
from torch_geometric_temporal.nn.attention.mtgnn import MTGNN
from pygtemporal_models.stemgnn import StemGNN
from pygtemporal_models.agcrn import AGCRN
from pygtemporal_models.fouriergnn import FGN
from pygtemporal_models.graphwavenet import gwnet
from pygtemporal_models.math_utils import weighted_mse

def plot_node_errors(pred_np, label_np, epoch, save_dir):
    """
    Plot error heatmap for nodes in ENSO region
    
    Args:
        pred_np (np.ndarray): Predictions
        label_np (np.ndarray): Ground truth
        enso_indices (np.ndarray): Indices of ENSO region nodes
        epoch (int): Current epoch
        save_dir (str): Directory to save plots
    """
    # Calculate error for each node
    errors = np.abs(pred_np - label_np)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot error heatmap
    sns.heatmap(errors, cmap='YlOrRd', cbar_kws={'label': 'Absolute Error'})
    plt.title(f'Node-wise Errors in ENSO Region (Epoch {epoch})')
    plt.xlabel('Time Steps')
    plt.ylabel('Nodes')
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'node_errors_epoch_{epoch}.png'))
    plt.close()

def compute_fourier_loss(pred, target):
    """
    Compute loss based on DFT of predictions and targets
    
    Args:
        pred (torch.Tensor): Predictions of shape (B, N, T)
        target (torch.Tensor): Targets of shape (B, N, T)
        
    Returns:
        torch.Tensor: Fourier loss
    """
    # Compute DFT along temporal dimension
    pred_fft = fft.fft(pred, dim=-1)
    target_fft = fft.fft(target, dim=-1)
    
    # Compute magnitude spectrum
    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)
    
    # Compute MSE between magnitude spectra
    fourier_loss = torch.mean((pred_magnitude - target_magnitude) ** 2)
    
    return fourier_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/wrapped_grid_graph.pt",
                       help="Path to graph file for ENSO dataset")
    parser.add_argument("--model_name", type=str, default="graphode",
                       choices=["node", "graphode", "mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"],
                       help="Model to use for forecasting")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden size of the model")
    parser.add_argument("--window", type=int, default=6,
                       help="Number of timesteps to use as input")
    parser.add_argument("--horizon", type=int, default=12,
                       help="Number of timesteps to forecast")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Number of epochs to train for")
    parser.add_argument("--patience", type=int, default=50,
                       help="Patience for early stopping")
    parser.add_argument("--use_normalization", action="store_true", default=True,
                       help="Whether to normalize the data")
    parser.add_argument("--use_loss_weights", action="store_true", default=True,
                       help="Whether to use weighted loss function")
    parser.add_argument("--train_length", type=int, default=700,
                       help="Number of time steps to use for training")
    parser.add_argument("--ode_encoder_decoder", action="store_true",
                       help="Use Neural ODE encoder-decoder structure")
    parser.add_argument("--graph_encoder", type=str, default="gcn", choices=["gcn", "gat"],
                       help="Graph encoder to use (GCN or GAT)")
    parser.add_argument("--use_periodic_activation", action="store_true",
                       help="Use periodic activation")
    parser.add_argument("--use_region_only", action="store_true",
                       help="Use only nodes in the ENSO region")
    parser.add_argument("--add_sin_cos", action="store_true",
                       help="Add sin and cos to the input")
    parser.add_argument("--use_region_data", action="store_true", default=True,
                        help="Use only region data for evaluation (default=True)")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples for NSDE")
    parser.add_argument("--from_checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Whether to save the best model during training")
    parser.add_argument("--gnn_latent_dim", type=int, default=None,
                        help="Latent dimension for GNN layers")
    # Additional arguments for pygtemporal models
    parser.add_argument("--multi_layer", type=int, default=5,
                       help="Number of layers for StemGNN")
    parser.add_argument("--input_dim", type=int, default=1,
                       help="Input dimension for models")
    parser.add_argument("--output_dim", type=int, default=1,
                       help="Output dimension for models")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of layers for some models")
    parser.add_argument("--cheb_k", type=int, default=2,
                       help="Chebyshev filter size for some models")
    parser.add_argument("--rnn_units", type=int, default=64,
                       help="RNN units for AGCRN")
    parser.add_argument("--embed_dim", type=int, default=32,
                       help="Embedding dimension for some models")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="Directory to save results and visualizations")
    parser.add_argument("--plot_interval", type=int, default=10,
                       help="Interval (in epochs) for plotting node errors")
    parser.add_argument("--use_skip_connection", action="store_true",
                       help="Use skip connections in GraphODE model")
    parser.add_argument("--use_fourier_loss", action="store_true",
                       help="Use Fourier-based loss in addition to MSE")
    parser.add_argument("--fourier_lambda", type=float, default=0.1,
                       help="Weight for Fourier loss term")

    args = parser.parse_args()
    # Default to True for use_region_data if not specified
    if not hasattr(args, 'use_region_data') or args.use_region_data is None:
        args.use_region_data = True

    print(f"args.use_region_data: {args.use_region_data}")
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data using graph_dataset_enso
    print(f"Loading graph data from {args.input_file}...")
    dataloader = OceanGraphDataset(
        graph_file=args.input_file,
        use_normalization=args.use_normalization,
        use_region_only=args.use_region_only,
        train_length=args.train_length
    )
    
    # Get dataset for training and testing
    train_dataset, test_dataset = dataloader.get_dataset(window=args.window, 
                                                        horizon=args.horizon)
    
    # Extract edge_index for graph models and ensure it's on the correct device
    edge_index = dataloader.edge_index.to(device)
    
    # Number of nodes in the graph
    grid_size = dataloader.time_series.shape[0]
    print(f"Graph dataset with {grid_size} nodes")
    print(f"ENSO region contains {dataloader.enso_mask.sum().item()} nodes")
    
    # Get the indices of nodes in the ENSO region for evaluation
    enso_indices = dataloader.get_enso_indices()
    print(f"enso_indices.shape: {enso_indices.shape}")

    enso_mask = dataloader.get_enso_mask()
    
    # Prepare data
    train_input = np.array(train_dataset.features)
    train_target = np.array(train_dataset.targets)
    test_input = np.array(test_dataset.features)
    test_target = np.array(test_dataset.targets)

    # Create data loaders - for graph models, we keep the tensor shape as (B, N, T)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(device)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)
    
    # For NODE model, we need to unsqueeze the channel dimension
    if args.model_name == "node":
        train_x_tensor = train_x_tensor.unsqueeze(1)
        train_target_tensor = train_target_tensor.unsqueeze(1)
    
    # For PyG temporal models, we need to unsqueeze the channel dimension
    if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
        train_x_tensor = train_x_tensor.unsqueeze(1)  # (B, 1, N, T)
        train_target_tensor = train_target_tensor.unsqueeze(1)  # (B, 1, N, T)
    
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, 
                                             shuffle=True, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)
    
    # For NODE model, we need to unsqueeze the channel dimension
    if args.model_name == "node":
        test_x_tensor = test_x_tensor.unsqueeze(1)
        test_target_tensor = test_target_tensor.unsqueeze(1)
    
    # For PyG temporal models, we need to unsqueeze the channel dimension
    if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
        test_x_tensor = test_x_tensor.unsqueeze(1)  # (B, 1, N, T)
        test_target_tensor = test_target_tensor.unsqueeze(1)  # (B, 1, N, T)
    
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=True)

    input_dim = grid_size
    print(f"Input dimension: {input_dim}")

    # Set reduced hidden size if using GAT to save memory
    if args.model_name == "graphode" and args.graph_encoder == "gat" and args.hidden_size > 64:
        print(f"Reducing hidden size from {args.hidden_size} to 64 for GAT to save memory")
        args.hidden_size = 64

    # Reduce batch size for GAT models to save memory
    if args.model_name == "graphode" and args.graph_encoder == "gat" and args.batch_size > 32:
        original_batch_size = args.batch_size
        args.batch_size = 16
        print(f"Reducing batch size from {original_batch_size} to {args.batch_size} for GAT to save memory")

    # Model initialization
    if args.model_name == "node":
        if args.ode_encoder_decoder:
            model = NeuralODEForecaster(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon,
                use_periodic_activation=args.use_periodic_activation
            ).to(device)
        else:
            model = TimeSeriesNODE(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon,
                use_periodic_activation=args.use_periodic_activation
            ).to(device)
        print(f"Initialized NODE model with input dimension {input_dim}")
    elif args.model_name == "graphode":
        if args.ode_encoder_decoder:
            model = NeuralGDEForecaster(
                input_dim=1,  # For graph models, node feature dim is 1
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon,
                num_nodes=grid_size,
                use_periodic_activation=args.use_periodic_activation,
                graph_encoder=args.graph_encoder,
                gnn_latent_dim=args.gnn_latent_dim,
                use_skip_connection=args.use_skip_connection
            ).to(device)
        else:
            model = GraphNeuralODE(
                node_features=1,  # For graph models, node feature dim is 1
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon,
                use_periodic_activation=args.use_periodic_activation,
                graph_encoder=args.graph_encoder,
                gnn_latent_dim=args.gnn_latent_dim,
                use_skip_connection=args.use_skip_connection
            ).to(device)
        print(f"Initialized GraphODE model with {grid_size} nodes using {args.graph_encoder} encoder")
        if args.gnn_latent_dim:
            print(f"Using GNN latent dimension of {args.gnn_latent_dim}")
        if args.use_skip_connection:
            print("Using skip connections in GraphODE model")
    elif args.model_name == "mtgnn":
        model = MTGNN(gcn_true=True, 
                      build_adj=True, 
                      gcn_depth=3, 
                      num_nodes=grid_size, 
                      kernel_set=[1,1,1,1], 
                      kernel_size=1, 
                      dropout=0.3, 
                      subgraph_size=grid_size, 
                      node_dim=1,
                      dilation_exponential=1,
                      conv_channels=32, 
                      residual_channels=32, 
                      skip_channels=128, 
                      end_channels=128, 
                      seq_length=args.window, 
                      in_dim=1, 
                      out_dim=args.horizon, 
                      layers=3, 
                      propalpha=0.05, 
                      tanhalpha=3, 
                      layer_norm_affline=True).to(device)
        print(f"Initialized MTGNN model with {grid_size} nodes")
    elif args.model_name == "stemgnn":
        model = StemGNN(units=grid_size,
                      stack_cnt=2,
                      time_step=args.window,
                      multi_layer=args.multi_layer,
                      horizon=args.horizon).to(device)
        print(f"Initialized StemGNN model with {grid_size} nodes")
    elif args.model_name == "agcrn":
        model = AGCRN(args=args, num_nodes=grid_size).to(device)
        print(f"Initialized AGCRN model with {grid_size} nodes")
    elif args.model_name == "fgnn":
        model = FGN(pre_length=args.horizon, 
                    embed_size=args.embed_dim, 
                    feature_size=grid_size, 
                    seq_length=args.window, 
                    hidden_size=args.rnn_units).to(device)
        print(f"Initialized FGN model with {grid_size} nodes")
    elif args.model_name == "wavenet":
        model = gwnet(device=device, 
                    window=args.window, 
                    horizon=args.horizon,
                    num_nodes=grid_size,
                    in_dim=args.input_dim,
                    out_dim=args.horizon).to(device)
        print(f"Initialized Graph WaveNet model with {grid_size} nodes")

    # Load model from checkpoint if provided
    if args.from_checkpoint and os.path.exists(args.from_checkpoint):
        print(f"Loading model from checkpoint: {args.from_checkpoint}")
        checkpoint = torch.load(args.from_checkpoint)
        if isinstance(checkpoint, tuple):
            model_state = checkpoint[0]
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(checkpoint)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_loss = np.inf
    patience_counter = 0

    losses_train, losses_test = [], []
    rmses_train, rmses_test = [], []
    rmses_train_reconstructed, rmses_test_reconstructed = [], []

    # Convert normalization parameters to device tensors if needed later
    max_tensor = torch.tensor(dataloader._max, device=device)
    min_tensor = torch.tensor(dataloader._min, device=device)
    if hasattr(dataloader, '_std'):
        std_tensor = torch.tensor(dataloader._std, device=device)
    else:
        std_tensor = None

    # CPU versions for numpy operations
    max = dataloader._max
    min = dataloader._min
    best_model = model

    # Training loop
    print(f"Training {args.model_name} model...")
    for epoch in tqdm(range(args.epochs)):
        # Training
        model.train()
        train_losses = []
        train_rmses = []
        train_rmses_recon = []

        # print train loader length
        for encoder_input, label in train_loader:
            # Ensure inputs are on the correct device
            encoder_input = encoder_input.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.model_name == "graphode":
                output = model(encoder_input, edge_index=edge_index)
            elif args.model_name == "node":
                output = model(encoder_input.squeeze(1))
            elif args.model_name == "stemgnn":
                output, _ = model(encoder_input)
            elif args.model_name in ["mtgnn", "agcrn", "fgnn", "wavenet"]:
                output = model(encoder_input).permute(0, 3, 2, 1)  # Adjust output dimensions to match label
                
            # Compute loss
            if args.use_loss_weights:
                if std_tensor is not None:
                    if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                        # For PyG temporal models, use weighted_mse directly
                        if args.add_sin_cos:
                            output_trim = output[:, :, :-2, :]
                            label_trim = label[:, :, :-2, :]
                            std_trim = std_tensor[:-2]
                            loss = weighted_mse(label_trim, output_trim, std_trim)
                        else:
                            loss = weighted_mse(label, output, std_tensor)
                    else:
                        # For NODE/GraphODE models use their compute_loss method
                        loss = model.compute_loss(output, label, std_tensor, add_sin_cos=args.add_sin_cos)
                else:
                    if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                        # If no std_tensor available, use regular MSE
                        loss = torch.nn.MSELoss()(output, label)
                    else:
                        loss = model.compute_loss(output, label)
            else:
                # Without weighting
                if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                    loss = torch.nn.MSELoss()(output, label)
                else:
                    loss = model.compute_loss(output, label)
            
            # Add Fourier loss if enabled
            if args.use_fourier_loss:
                # Ensure we're using the right dimensions for Fourier transform
                if args.model_name == "node":
                    pred_for_fft = output.squeeze(1)  # Remove channel dimension
                    target_for_fft = label.squeeze(1)
                else:
                    pred_for_fft = output.squeeze(1)  # Remove channel dimension
                    target_for_fft = label.squeeze(1)
                
                fourier_loss = compute_fourier_loss(pred_for_fft, target_for_fft)
                loss = loss + args.fourier_lambda * fourier_loss
                
            loss.backward()
            optimizer.step()

            # Log both losses if Fourier loss is enabled
            if args.use_fourier_loss:
                train_losses.append((loss.item(), fourier_loss.item()))
            else:
                train_losses.append(loss.item())

            # Compute metrics
            train_rmses.append(loss.item())

            # Convert to numpy for metric calculation
            if args.model_name == "node":
                pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
            else: # all graph models are the same
                pred_np = output.squeeze().detach().cpu().numpy()
                label_np = label.squeeze().detach().cpu().numpy()

            if args.use_normalization:
                label_np = inverse_normalize(label_np, max, min)
                pred_np = inverse_normalize(pred_np, max, min)
            
            # Only calculate RMSE on the ENSO region for reporting
            if args.use_region_data and not args.use_region_only:
                # Convert enso_indices to numpy
                enso_indices_np = enso_indices.cpu().numpy()
                # Filter predictions and labels to only include ENSO region
                pred_np_enso = pred_np[:, enso_indices_np, :]
                label_np_enso = label_np[:, enso_indices_np, :]
                
                # Calculate RMSE on ENSO region
                rmse = np.sqrt(np.mean((label_np_enso - pred_np_enso)**2))
                
                # For ENSO dataset, use spatial average of ENSO region as the reconstruction metric
                nino34 = pred_np_enso.mean(axis=1)
                nino34_pred = label_np_enso.mean(axis=1)
            else:
                # Calculate RMSE on entire grid or already filtered dataset
                rmse = np.sqrt(np.mean((label_np - pred_np)**2))
                
                # For ENSO dataset, use spatial average as the reconstruction metric
                nino34 = pred_np.mean(axis=1)
                nino34_pred = label_np.mean(axis=1)
                
            rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
            train_rmses.append(rmse)
            train_rmses_recon.append(rmse_recon)

        # Validation
        model.eval()
        test_losses = []
        test_rmses = []
        test_rmses_recon = []
        best_enso_reconstructed_loss = np.inf

        with torch.no_grad():
            for encoder_input, label in test_loader:
                # Ensure inputs are on the correct device
                encoder_input = encoder_input.to(device)
                label = label.to(device)
                
                # Forward pass
                if args.model_name == "graphode":
                    output = model(encoder_input, edge_index=edge_index)
                elif args.model_name == "node":
                    output = model(encoder_input.squeeze(1))
                elif args.model_name == "stemgnn":
                    output, _ = model(encoder_input)
                elif args.model_name in ["mtgnn", "agcrn", "fgnn", "wavenet"]:
                    output = model(encoder_input).permute(0, 3, 2, 1)  # Adjust output dimensions to match label
                
                # Compute loss
                if args.use_loss_weights:
                    if std_tensor is not None:
                        if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                            # For PyG temporal models, use weighted_mse directly
                            if args.add_sin_cos:
                                output_trim = output[:, :, :-2, :]
                                label_trim = label[:, :, :-2, :]
                                std_trim = std_tensor[:-2]
                                loss = weighted_mse(label_trim, output_trim, std_trim)
                            else:
                                loss = weighted_mse(label, output, std_tensor)
                        else:
                            # For NODE/GraphODE models use their compute_loss method
                            loss = model.compute_loss(output, label, std_tensor, add_sin_cos=args.add_sin_cos)
                    else:
                        if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                            # If no std_tensor available, use regular MSE
                            loss = torch.nn.MSELoss()(output, label)
                        else:
                            loss = model.compute_loss(output, label)
                else:
                    # Without weighting
                    if args.model_name in ["mtgnn", "stemgnn", "agcrn", "fgnn", "wavenet"]:
                        loss = torch.nn.MSELoss()(output, label)
                    else:
                        loss = model.compute_loss(output, label)
                
                # Add Fourier loss if enabled
                if args.use_fourier_loss:
                    # Ensure we're using the right dimensions for Fourier transform
                    if args.model_name == "node":
                        pred_for_fft = output.squeeze(1)  # Remove channel dimension
                        target_for_fft = label.squeeze(1)
                    else:
                        pred_for_fft = output.squeeze(1)  # Remove channel dimension
                        target_for_fft = label.squeeze(1)
                    
                    fourier_loss = compute_fourier_loss(pred_for_fft, target_for_fft)
                    loss = loss + args.fourier_lambda * fourier_loss

                # Log both losses if Fourier loss is enabled
                if args.use_fourier_loss:
                    test_losses.append((loss.item(), fourier_loss.item()))
                else:
                    test_losses.append(loss.item())

                # Convert to numpy for metric calculation
                if args.model_name == "node":
                    pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                    label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
                else:
                    pred_np = output.squeeze().detach().cpu().numpy()
                    label_np = label.squeeze().detach().cpu().numpy()
                
                if args.use_normalization:
                    label_np = inverse_normalize(label_np, max, min)
                    pred_np = inverse_normalize(pred_np, max, min)
                
                # Only calculate RMSE on the ENSO region for reporting
                if args.use_region_data and not args.use_region_only:
                    # Convert enso_indices to numpy
                    enso_indices_np = enso_indices.cpu().numpy()
                    # Filter predictions and labels to only include ENSO region
                    pred_np_enso = pred_np[:, enso_indices_np, :]
                    label_np_enso = label_np[:, enso_indices_np, :]
                    
                    # Calculate RMSE on ENSO region
                    rmse = np.sqrt(np.mean((label_np_enso - pred_np_enso)**2))
                    
                    # For ENSO dataset, use spatial average of ENSO region as the reconstruction metric
                    nino34 = pred_np_enso.mean(axis=1)
                    nino34_pred = label_np_enso.mean(axis=1)
                else:
                    # Calculate RMSE on entire grid or already filtered dataset
                    rmse = np.sqrt(np.mean((label_np - pred_np)**2))
                    
                    # For ENSO dataset, use spatial average as the reconstruction metric
                    nino34 = pred_np.mean(axis=1)
                    nino34_pred = label_np.mean(axis=1)
                    
                rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
                test_rmses.append(rmse)
                test_rmses_recon.append(rmse_recon)

                # Plot node errors every plot_interval epochs
                if epoch % args.plot_interval == 0 and args.use_region_data:
                    if args.model_name == "node":
                        pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                        label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
                    else:
                        pred_np = output.squeeze().detach().cpu().numpy()
                        label_np = label.squeeze().cpu().numpy()
                    
                    if args.use_normalization:
                        label_np = inverse_normalize(label_np, max, min)
                        pred_np = inverse_normalize(pred_np, max, min)
                    
                    # Get ENSO region predictions
                    
                    enso_indices_np = enso_indices.cpu().numpy()
                    if not args.use_region_only:
                        pred_np_enso = pred_np[:, enso_indices_np, :]
                        label_np_enso = label_np[:, enso_indices_np, :]
                    else:
                        pred_np_enso = pred_np
                        label_np_enso = label_np
                    
                    # Plot errors
                    plot_node_errors(pred_np_enso, label_np_enso, epoch, args.save_dir)

        # Log metrics
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        train_rmse = np.mean(train_rmses)
        test_rmse = np.mean(test_rmses)
        train_rmse_recon = np.mean(train_rmses_recon)
        test_rmse_recon = np.mean(test_rmses_recon)

        losses_train.append(train_loss)
        losses_test.append(test_loss)
        rmses_train.append(train_rmse)
        rmses_test.append(test_rmse)
        rmses_train_reconstructed.append(train_rmse_recon)
        rmses_test_reconstructed.append(test_rmse_recon)

        if args.use_fourier_loss:
            train_loss = np.mean([l[0] for l in train_losses])
            train_fourier_loss = np.mean([l[1] for l in train_losses])
            test_loss = np.mean([l[0] for l in test_losses])
            test_fourier_loss = np.mean([l[1] for l in test_losses])
            
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Fourier Loss: {train_fourier_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Fourier Loss: {test_fourier_loss:.4f}")
        else:
            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}")
        # Early stopping
        if test_rmse_recon < best_enso_reconstructed_loss:
            best_enso_reconstructed_loss = test_rmse_recon
            patience_counter = 0
            best_model = model
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best test RMSE reconstructed: {best_enso_reconstructed_loss:.3f}")
                break

    # Save results
    # Convert enso_indices to numpy for save_results
    indsst = enso_indices.cpu().numpy() if args.use_region_data else None
    save_results(args, best_model, test_x_tensor, test_target_tensor, test_dataset_new, max, min,
                losses_train, losses_test, rmses_train, rmses_test,
                rmses_train_reconstructed, rmses_test_reconstructed, 
                edge_index=edge_index, indsst=indsst)
    
    # Save the best model if requested
    if args.save_best_model:
        model_dir = os.path.join(args.save_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{args.model_name}_{args.graph_encoder}_best.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
