import torch
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from utils_pca import reconstruct_enso

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/wrapped_grid_graph.pt",
                       help="Path to graph file for ENSO dataset")
    parser.add_argument("--model_name", type=str, default="graphode",
                       choices=["node", "graphode"],
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
    parser.add_argument("--use_normalization", action="store_true",
                       help="Whether to normalize the data")
    parser.add_argument("--use_loss_weights", action="store_true",
                       help="Whether to use weighted loss function")
    parser.add_argument("--train_length", type=int, default=700,
                       help="Number of time steps to use for training")
    parser.add_argument("--ode_encoder_decoder", action="store_true",
                       help="Use Neural ODE encoder-decoder structure")
    parser.add_argument("--use_periodic_activation", action="store_true",
                       help="Use periodic activation")
    parser.add_argument("--use_region_only", action="store_true",
                       help="Use only nodes in the ENSO region")
    parser.add_argument("--add_sin_cos", action="store_true",
                       help="Add sin and cos to the input")
    parser.add_argument("--use_region_data", action="store_true",
                        help="Use only region data for evaluation (default=True)")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples for NSDE")

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
    
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, 
                                             shuffle=True, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)
    
    # For NODE model, we need to unsqueeze the channel dimension
    if args.model_name == "node":
        test_x_tensor = test_x_tensor.unsqueeze(1)
        test_target_tensor = test_target_tensor.unsqueeze(1)
    
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=True)

    input_dim = grid_size
    print(f"Input dimension: {input_dim}")

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
                use_periodic_activation=args.use_periodic_activation
            ).to(device)
        else:
            model = GraphNeuralODE(
                node_features=1,  # For graph models, node feature dim is 1
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon,
                use_periodic_activation=args.use_periodic_activation
            ).to(device)
        print(f"Initialized GraphODE model with {grid_size} nodes")

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
        print(f"Train loader length: {len(train_loader)}")
        for encoder_input, label in train_loader:
            # Ensure inputs are on the correct device
            encoder_input = encoder_input.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.model_name == "graphode":
                output = model(encoder_input, edge_index=edge_index)
            else:  # node model
                output = model(encoder_input.squeeze(1))
                
            # Compute loss
            if args.use_loss_weights:
                if std_tensor is not None:
                    loss = model.compute_loss(output, label, std_tensor, add_sin_cos=args.add_sin_cos)
                else:
                    loss = model.compute_loss(output, label)
            else:
                loss = model.compute_loss(output, label)
                
            loss.backward()
            optimizer.step()
            # print(f"loss: {loss.item()}")

            # Compute metrics
            train_losses.append(loss.item())

            # Convert to numpy for metric calculation
            if args.model_name == "node":
                pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
            else:
                pred_np = output.detach().cpu().numpy()
                label_np = label.detach().cpu().numpy()
            
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
                else:  # node model
                    output = model(encoder_input.squeeze(1))
                
                # Compute loss
                if args.use_loss_weights:
                    if std_tensor is not None:
                        loss = model.compute_loss(output, label, std_tensor, add_sin_cos=args.add_sin_cos)
                    else:
                        loss = model.compute_loss(output, label)
                else:
                    loss = model.compute_loss(output, label)

                test_losses.append(loss.item())

                # Convert to numpy for metric calculation
                if args.model_name == "node":
                    pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                    label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
                else:
                    pred_np = output.detach().cpu().numpy()
                    label_np = label.detach().cpu().numpy()
                
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

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Train RMSE Recon: {train_rmse_recon:.4f}, Test RMSE Recon: {test_rmse_recon:.4f}")

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
    print(f"enso_indices.shape: {enso_indices.shape}")
    indsst = enso_indices.cpu().numpy() if args.use_region_data else None
    print(f"indsst.shape: {indsst.shape}")
    save_results(args, best_model, test_x_tensor, test_target_tensor, test_dataset_new, max, min,
                losses_train, losses_test, rmses_train, rmses_test,
                rmses_train_reconstructed, rmses_test_reconstructed, 
                edge_index=edge_index, indsst=indsst)

if __name__ == "__main__":
    main()
