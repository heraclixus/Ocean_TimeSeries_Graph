from pygtemporal_models.pyg_temp_dataset import (
    SSTDatasetLoader, 
    inverse_normalize, 
    batch_data_to_timeseries,
    stochastic_batch_data_to_timeseries
)
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils_pca import reconstruct_enso
# Import baseline models
from baseline_models.node import TimeSeriesNODE, NeuralODEForecaster
from baseline_models.ncde import TimeSeriesCDE
from baseline_models.nsde import TimeSeriesSDE, NeuralSDEForecaster
from baseline_models.graphode import GraphNeuralODE
from baseline_models.kalman_filter import KalmanForecaster
from baseline_models.dmd import DMDForecast
from baseline_models.gaussian_process import TimeSeriesGP
from baseline_models.koopman import DeepKoopman
from baseline_models.utils import save_results
from baseline_models.arima import MultiARIMA
from baseline_models.arimax import ARIMAX
from baseline_models.garch import MultiGARCH



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/sst_pcs.mat")
    parser.add_argument("--model_name", type=str, default="node",
                       choices=["node", "ncde", "nsde", "graphode", "kalman", 
                               "dmd", "gp", "koopman", "arima", "arimax", "garch"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n_pcs", type=int, default=20)
    parser.add_argument("--use_normalization", action="store_true")
    parser.add_argument("--use_loss_weights", action="store_true")
    parser.add_argument("--add_sin_cos", action="store_true",
                       help="Add sine and cosine features with period 12")
    parser.add_argument("--train_length", type=int, default=700,
                       help="Number of time steps to use for training")
    parser.add_argument("--ode_encoder_decoder", action="store_true",
                       help="Use Neural ODE encoder-decoder structure")
    parser.add_argument("--n_samples", type=int, default=20,
                       help="Number of sample paths for NSDE")

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    sst_dataloader = SSTDatasetLoader(filepath=args.input_file, 
                                     use_normalization=args.use_normalization,
                                     n_pcs=args.n_pcs,
                                     add_sin_cos=args.add_sin_cos,
                                     train_length=args.train_length)

    train_dataset, test_dataset = sst_dataloader.get_dataset(window=args.window, 
                                                           horizon=args.horizon)
    
    # Prepare data
    train_input = np.array(train_dataset.features)
    train_target = np.array(train_dataset.targets)
    test_input = np.array(test_dataset.features)
    test_target = np.array(test_dataset.targets)

    # Create data loaders
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).unsqueeze(1).to(device)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).unsqueeze(1).to(device)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=args.batch_size, 
                                             shuffle=False, drop_last=True)

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).unsqueeze(1).to(device)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).unsqueeze(1).to(device)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, 
                                            shuffle=False, drop_last=True)

    
    if args.add_sin_cos:
        input_dim = args.n_pcs + 2 
    else:
        input_dim = args.n_pcs
    
    # Model initialization
    if args.model_name == "node":
        if args.ode_encoder_decoder:
            model = NeuralODEForecaster(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon
            ).to(device)
        else:
            model = TimeSeriesNODE(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon
            ).to(device)
    elif args.model_name == "ncde":
        model = TimeSeriesCDE(
            input_dim=input_dim,
            hidden_dim=args.hidden_size,
            forecast_horizon=args.horizon
        ).to(device)
    elif args.model_name == "nsde":
        if args.ode_encoder_decoder:
            model = NeuralSDEForecaster(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                time_series_length=args.window,
                forecast_length=args.horizon
            ).to(device)
        else:
            model = TimeSeriesSDE(
                input_dim=input_dim,
                hidden_dim=args.hidden_size,
                forecast_horizon=args.horizon
            ).to(device)
    elif args.model_name == "graphode":
        model = GraphNeuralODE(
            node_features=1,
            hidden_dim=args.hidden_size,
            forecast_horizon=args.horizon
        ).to(device)
    elif args.model_name == "kalman":
        model = KalmanForecaster(
            input_dim=input_dim,
            filter_type='extended'
        ).to(device)
    elif args.model_name == "dmd":
        model = DMDForecast(
            input_dim=input_dim,
            rank=args.n_pcs//2
        )
    elif args.model_name == "gp":
        model = TimeSeriesGP(
            input_dim=input_dim,
            forecast_horizon=args.horizon
        )
        # Store dataloader for normalization
        model.dataloader = sst_dataloader
        model.add_sin_cos = args.add_sin_cos
    elif args.model_name == "koopman":
        model = DeepKoopman(
            input_dim=input_dim,
            latent_dim=args.hidden_size,
            hidden_dims=[args.hidden_size, args.hidden_size]
        ).to(device)
    elif args.model_name == "arima":
        model = MultiARIMA(order=(1,1,1))
        model.add_sin_cos = args.add_sin_cos
    elif args.model_name == "arimax":
        model = ARIMAX(order=(1,1))
        model.add_sin_cos = args.add_sin_cos
    elif args.model_name == "garch":
        model = MultiGARCH(p=1, q=1)
        model.add_sin_cos = args.add_sin_cos

    # Handle non-neural models separately
    if args.model_name in ["dmd", "gp", "arima", "arimax", "garch"]:
        print(f"Fitting {args.model_name} model...")
        if args.model_name == "gp":
            # Pass both input and target for tracking RMSE during training
            model.fit(train_x_tensor.squeeze(1), train_target_tensor.squeeze(1), n_iter=args.epochs)
        else:
            model.fit(train_x_tensor.squeeze(1), n_iter=args.epochs)
        
        if args.model_name == "gp":
            predictions, std = model.predict(test_x_tensor.squeeze(1), return_std=True)
        elif args.model_name == "garch":
            predictions, volatility = model.predict(test_x_tensor.squeeze(1), args.horizon)
        else:
            predictions = model.predict(test_x_tensor.squeeze(1), args.horizon)

        # Compute and save results
        save_results(args, model, test_x_tensor, test_target_tensor, sst_dataloader)
        exit(0)

    # Training neural models
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_loss = np.inf
    patience_counter = 0

    losses_train, losses_test = [], []
    rmses_train, rmses_test = [], []
    rmses_train_reconstructed, rmses_test_reconstructed = [], []


    if args.add_sin_cos:
        max = sst_dataloader._max[:-2]
        min = sst_dataloader._min[:-2]
    else:
        max = sst_dataloader._max
        min = sst_dataloader._min

    print(f"Training {args.model_name} model...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        train_rmses = []
        train_rmses_recon = []

        for encoder_input, label in train_loader:
            optimizer.zero_grad()
            # Forward pass
            if args.model_name == "kalman":
                output, output_cov = model(encoder_input.squeeze(1), args.horizon)
                if args.add_sin_cos:
                    output = output[:, :-2, :]
                    label = label[:, :, :-2, :]
                if args.use_loss_weights:
                    loss = model.compute_loss(output, label, output_cov, sst_dataloader._std, args.add_sin_cos)
                else:
                    loss = model.compute_loss(output, label, output_cov, add_sin_cos=args.add_sin_cos)
            else:
                output = model(encoder_input.squeeze(1), n_samples=args.n_samples)
                if args.add_sin_cos:
                    if len(output.shape) == 4: # stochastic models
                        output = output[:, :, :-2, :]
                    else:
                        output = output[:, :-2, :]
                    label = label[:, :, :-2, :].squeeze(1)
                if args.use_loss_weights:
                    loss = model.compute_loss(output, label, sst_dataloader._std, args.add_sin_cos)
                else:
                    loss = model.compute_loss(output, label, add_sin_cos=args.add_sin_cos)

            loss.backward()
            optimizer.step()

            # Compute metrics
            train_losses.append(loss.item())

            if args.model_name == "nsde":
                pred_np = stochastic_batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
                pred_np = np.mean(pred_np, axis=0)
            else:
                pred_np = batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
            label_np = batch_data_to_timeseries(label.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
            if args.use_normalization:
                label_np = inverse_normalize(label_np, max, min)
                pred_np = inverse_normalize(pred_np, max, min)
            
            rmse = np.sqrt(np.mean((label_np - pred_np)**2))
            train_rmses.append(rmse)

            # Compute reconstructed RMSE
            nino34, nino34_pred = reconstruct_enso(pcs=pred_np, real_pcs=label_np, 
                                                 top_n_pcs=args.n_pcs, flag="train")
            rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
            train_rmses_recon.append(rmse_recon)

        # Validation
        model.eval()
        test_losses = []
        test_rmses = []
        test_rmses_recon = []

        with torch.no_grad():
            for encoder_input, label in test_loader:
                if args.model_name == "kalman":
                    output, output_cov = model(encoder_input.squeeze(1), args.horizon)
                    if args.add_sin_cos:
                        output = output[:, :-2, :]
                        label = label[:, :, :-2, :].squeeze(1)
                    if args.use_loss_weights:
                        loss = model.compute_loss(output, label, output_cov, sst_dataloader._std, args.add_sin_cos)
                    else:
                        loss = model.compute_loss(output, label, output_cov, add_sin_cos=args.add_sin_cos)
                elif args.model_name == "nsde":
                    output = model(encoder_input.squeeze(1), n_samples=args.n_samples)
                    if args.add_sin_cos:
                        output = output[:, :, :-2, :]
                        label = label[:, :, :-2, :].squeeze(1)
                    if args.use_loss_weights:
                        loss = model.compute_loss(output, label, sst_dataloader._std, args.add_sin_cos)
                    else:
                        loss = model.compute_loss(output, label, add_sin_cos=args.add_sin_cos)
                else:
                    output = model(encoder_input.squeeze(1))
                    if args.add_sin_cos:
                        if len(output.shape) == 4: # stochastic models
                            output = output[:, :, -2, :]
                        else:
                            output = output[:, :-2, :]
                        label = label[:, :, :-2, :].squeeze(1)
                    if args.use_loss_weights:
                        loss = model.compute_loss(output, label, sst_dataloader._std, args.add_sin_cos)
                    else:
                        loss = model.compute_loss(output, label, add_sin_cos=args.add_sin_cos)

                test_losses.append(loss.item())

                # Compute metrics
                if args.model_name == "nsde":
                    pred_np = stochastic_batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
                    pred_np = np.mean(pred_np, axis=0)
                else:
                    pred_np = batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
                label_np = batch_data_to_timeseries(label.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
                if args.use_normalization:
                    label_np = inverse_normalize(label_np, max, min)
                    pred_np = inverse_normalize(pred_np, max, min)
                
                rmse = np.sqrt(np.mean((label_np - pred_np)**2))
                test_rmses.append(rmse)

                nino34, nino34_pred = reconstruct_enso(pcs=pred_np, real_pcs=label_np, top_n_pcs=args.n_pcs, flag="test")
                rmse_recon = np.sqrt(np.mean((nino34-nino34_pred)**2))
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
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save results
    save_results(args, model, test_x_tensor, test_target_tensor, sst_dataloader, max, min,
                losses_train, losses_test, rmses_train, rmses_test,
                rmses_train_reconstructed, rmses_test_reconstructed)

