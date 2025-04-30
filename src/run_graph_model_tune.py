import torch
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
from utils_pca import reconstruct_enso

# Import the dataset class
from pygtemporal_models.graph_dataset_enso import (
    OceanGraphDataset,
    inverse_normalize
)

# Import models
from baseline_models.node import TimeSeriesNODE, NeuralODEForecaster
from baseline_models.graphode import GraphNeuralODE, NeuralGDEForecaster
from pygtemporal_models.pyg_temp_dataset import batch_data_to_timeseries

# Global tracking of results
all_results = []

def train_graph_model(config, checkpoint_dir=None, args=None):
    """
    Training function for ray tune
    
    Args:
        config: hyperparameters from ray tune
        checkpoint_dir: directory for checkpoints
        args: command line arguments
    """
    # Update args with config parameters
    for key, value in config.items():
        setattr(args, key, value)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data using graph_dataset_enso
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
    
    # Get the indices of nodes in the ENSO region for evaluation
    enso_indices = dataloader.get_enso_indices()
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

    # Automatically reduce batch size for GAT to save memory
    if args.model_name == "graphode" and args.graph_encoder == "gat" and args.batch_size > 32:
        args.batch_size = 16

    # Model initialization
    if args.model_name == "node":
        if config.get("ode_encoder_decoder", args.ode_encoder_decoder):
            model = NeuralODEForecaster(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon,
                use_periodic_activation=config.get("use_periodic_activation", args.use_periodic_activation)
            ).to(device)
        else:
            model = TimeSeriesNODE(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon,
                use_periodic_activation=config.get("use_periodic_activation", args.use_periodic_activation)
            ).to(device)
    elif args.model_name == "graphode":
        if config.get("ode_encoder_decoder", args.ode_encoder_decoder):
            model = NeuralGDEForecaster(
                input_dim=1,  # For graph models, node feature dim is 1
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon,
                num_nodes=grid_size,
                use_periodic_activation=config.get("use_periodic_activation", args.use_periodic_activation),
                graph_encoder=config.get("graph_encoder", args.graph_encoder),
                gnn_latent_dim=config.get("gnn_latent_dim", None)
            ).to(device)
        else:
            model = GraphNeuralODE(
                node_features=1,  # For graph models, node feature dim is 1
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon,
                use_periodic_activation=config.get("use_periodic_activation", args.use_periodic_activation),
                graph_encoder=config.get("graph_encoder", args.graph_encoder),
                gnn_latent_dim=config.get("gnn_latent_dim", None)
            ).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_test_rmse_recon = float('inf')
    for epoch in range(args.epochs_per_tune):
        # Training
        model.train()
        train_losses = []
        train_rmses_recon = []

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
                if hasattr(dataloader, '_std'):
                    std_tensor = torch.tensor(dataloader._std, device=device)
                    loss = model.compute_loss(output, label, std_tensor, add_sin_cos=args.add_sin_cos)
                else:
                    loss = model.compute_loss(output, label)
            else:
                loss = model.compute_loss(output, label)
                
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Convert to numpy for metric calculation
            if args.model_name == "node":
                pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
                label_np = batch_data_to_timeseries(label.detach().cpu().squeeze(1).numpy())
            else:
                pred_np = output.detach().cpu().numpy()
                label_np = label.detach().cpu().numpy()
            
            # Inverse normalize if needed
            if args.use_normalization:
                label_np = inverse_normalize(label_np, dataloader._max, dataloader._min)
                pred_np = inverse_normalize(pred_np, dataloader._max, dataloader._min)
            
            # Only calculate RMSE on the ENSO region for reporting
            if args.use_region_data and not args.use_region_only:
                # Filter predictions and labels to only include ENSO region
                enso_indices_np = enso_indices.cpu().numpy()
                pred_np_enso = pred_np[:, enso_indices_np, :]
                label_np_enso = label_np[:, enso_indices_np, :]
                
                # For ENSO dataset, use spatial average of ENSO region as the reconstruction metric
                nino34 = pred_np_enso.mean(axis=1)
                nino34_pred = label_np_enso.mean(axis=1)
            else:
                # For ENSO dataset, use spatial average as the reconstruction metric
                nino34 = pred_np.mean(axis=1)
                nino34_pred = label_np.mean(axis=1)
                
            rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
            train_rmses_recon.append(rmse_recon)

        # Validation
        model.eval()
        test_losses = []
        test_rmses_recon = []

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
                    if hasattr(dataloader, '_std'):
                        std_tensor = torch.tensor(dataloader._std, device=device)
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
                
                # Inverse normalize if needed
                if args.use_normalization:
                    label_np = inverse_normalize(label_np, dataloader._max, dataloader._min)
                    pred_np = inverse_normalize(pred_np, dataloader._max, dataloader._min)
                
                # Only calculate RMSE on the ENSO region for reporting
                if args.use_region_data and not args.use_region_only:
                    # Filter predictions and labels to only include ENSO region
                    enso_indices_np = enso_indices.cpu().numpy()
                    pred_np_enso = pred_np[:, enso_indices_np, :]
                    label_np_enso = label_np[:, enso_indices_np, :]
                    
                    # For ENSO dataset, use spatial average of ENSO region as the reconstruction metric
                    nino34 = pred_np_enso.mean(axis=1)
                    nino34_pred = label_np_enso.mean(axis=1)
                else:
                    # For ENSO dataset, use spatial average as the reconstruction metric
                    nino34 = pred_np.mean(axis=1)
                    nino34_pred = label_np.mean(axis=1)
                    
                rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
                test_rmses_recon.append(rmse_recon)

        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        train_rmse_recon = np.mean(train_rmses_recon)
        test_rmse_recon = np.mean(test_rmses_recon)
        
        # Print metrics
        print(f"Epoch {epoch}: "
              f"test_rmse_recon={test_rmse_recon:.4f}, "
              f"train_rmse_recon={train_rmse_recon:.4f}, "
              f"test_loss={test_loss:.4f}, "
              f"train_loss={train_loss:.4f}")
        
        # Track best model locally
        if test_rmse_recon < best_test_rmse_recon:
            best_test_rmse_recon = test_rmse_recon
    
    # After all epochs, report metrics to tune for wandb integration
    # tune.report(
    #     test_rmse_recon=best_test_rmse_recon,
    #     train_rmse_recon=train_rmse_recon,
    #     test_loss=test_loss,
    #     train_loss=train_loss
    # )
    
    # Save the results to our global tracker
    global all_results
    result = {
        "config": config,
        "test_rmse_recon": best_test_rmse_recon,
        "train_rmse_recon": train_rmse_recon
    }
    
    print(f"Trial completed with best test RMSE: {best_test_rmse_recon:.4f}")
    
    # Add to global results
    all_results.append(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/home/mosaicml/Ocean_TimeSeries_Graph/data/wrapped_grid_graph.pt",
                       help="Path to graph file for ENSO dataset")
    parser.add_argument("--model_name", type=str, default="graphode",
                       choices=["node", "graphode"],
                       help="Model to use for forecasting")
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
    parser.add_argument("--epochs_per_tune", type=int, default=500,
                       help="Number of epochs to train for each tune trial")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of hyperparameter samples for Ray Tune")
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
    parser.add_argument("--add_sin_cos", action="store_true", default=False,
                       help="Add sin and cos to the input")
    parser.add_argument("--use_region_data", action="store_true", default=True,
                        help="Use only region data for evaluation (default=True)")
    parser.add_argument("--max_num_epochs", type=int, default=50,
                       help="Maximum number of epochs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=1,
                       help="Number of GPUs per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=8,
                       help="Number of CPUs per trial")
    parser.add_argument("--wandb_project", type=str, default="ocean_graph_ts",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_api_key", type=str, default="853c7fee8ce1ad1cf4cb1520d0fd750dce2e9b60",
                       help="Weights & Biases API key (optional if already logged in)")

    args = parser.parse_args()
    # Default to True for use_region_data if not specified
    if not hasattr(args, 'use_region_data') or args.use_region_data is None:
        args.use_region_data = True

    # Initialize Ray
    ray.init(num_cpus=args.cpus_per_trial * 2, num_gpus=2)
    
    # Define hyperparameter search space
    config = {
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "learning_rate": tune.choice([1e-5, 1e-4, 1e-3]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "ode_encoder_decoder": tune.choice([True, False]),
        "use_periodic_activation": tune.choice([True, False]),
        "use_region_only": tune.choice([True, False])
    }
    
    # Add model-specific hyperparameters
    if args.model_name == "graphode":
        # For GraphODE models, add GNN latent dimension as a hyperparameter
        config["gnn_latent_dim"] = tune.choice([16, 32, 64])
        config["graph_encoder"] = tune.choice(["gcn", "gat"])
    
    # Run trials with wandb integration using the logger callback
    experiment_name = f"{args.model_name}_{args.graph_encoder}_tuning"
    
    analysis = tune.run(
        tune.with_parameters(train_graph_model, args=args),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples,
        verbose=1,
        name=experiment_name,
        callbacks=[WandbLoggerCallback(
            project=args.wandb_project,
            api_key=args.wandb_api_key,
            log_config=True
        )]
    )
    
    # Find the best trial from our global tracker
    best_result = None
    for result in all_results:
        if best_result is None or result["test_rmse_recon"] < best_result["test_rmse_recon"]:
            best_result = result
    
    # If we found a best result, print it
    if best_result:
        print("\nBest Hyperparameters:")
        for key, value in best_result["config"].items():
            print(f"{key}: {value}")
            
        print(f"\nBest test RMSE: {best_result['test_rmse_recon']}")
        print(f"Train RMSE: {best_result.get('train_rmse_recon', 'N/A')}")
        
        # Apply the best hyperparameters to the original args
        args.hidden_size = best_result["config"]["hidden_size"]
        args.learning_rate = best_result["config"]["learning_rate"]
        args.batch_size = best_result["config"]["batch_size"]
        args.ode_encoder_decoder = best_result["config"]["ode_encoder_decoder"]
        args.use_periodic_activation = best_result["config"]["use_periodic_activation"]
        args.use_region_only = best_result["config"]["use_region_only"]
        
        if args.model_name == "graphode":
            args.gnn_latent_dim = best_result["config"].get("gnn_latent_dim", None)
            args.graph_encoder = best_result["config"].get("graph_encoder", "gcn")
    else:
        print("\nNo results found. Check if trials completed successfully.")

if __name__ == "__main__":
    main() 