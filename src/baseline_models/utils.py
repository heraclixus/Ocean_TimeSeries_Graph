import os
from pygtemporal_models.pyg_temp_dataset import (
    batch_data_to_timeseries, 
    inverse_normalize, 
    stochastic_batch_data_to_timeseries
)
from utils_pca import reconstruct_enso
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils_visualization_forecast import (
    plot_ts_channel_rmse, 
    plot_enso_anomaly_correlation, 
    plot_enso_anomaly_rmse, 
    plot_channel_rmse, 
    plot_enso_forecast_vs_real
)
import torch.nn as nn

class PeriodicActivation(nn.Module):
    def __init__(self, a=0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        return x + torch.sin(self.a * x) ** 2 / self.a


def plot_predictions(save_path, true_values, predicted_values, title):
    """
    Plot true vs predicted values
    
    Args:
        save_path (str): Directory to save the plot
        true_values (np.ndarray): True values
        predicted_values (np.ndarray): Predicted values
        title (str): Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label="True ENSO")
    plt.plot(predicted_values, label="Predicted ENSO")
    plt.xlabel("Time step")
    plt.ylabel("ENSO")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path, "enso_predictions.png"))
    plt.close()

def plot_prediction_with_uncertainty(save_path, true_values, predicted_samples, title):
    """
    Plot predictions with uncertainty bands
    
    Args:
        save_path (str): Directory to save the plot
        true_values (np.ndarray): True values
        predicted_samples (np.ndarray): Samples of predictions (shape: n_samples x time_steps)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 5))
    
    # Plot true values
    plt.plot(true_values, 'k-', label='True ENSO', zorder=3)
    
    # Calculate mean and percentiles of predictions
    mean_pred = np.mean(predicted_samples, axis=0)
    percentile_5 = np.percentile(predicted_samples, 5, axis=0)
    percentile_95 = np.percentile(predicted_samples, 95, axis=0)
    
    # Plot mean prediction and uncertainty band
    plt.plot(mean_pred, 'r-', label='Mean Prediction', zorder=2)
    plt.fill_between(range(len(mean_pred)), percentile_5, percentile_95,
                    color='r', alpha=0.2, label='90% Confidence Interval', zorder=1)
    
    plt.xlabel("Time step")
    plt.ylabel("ENSO")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path, "enso_predictions_with_uncertainty.png"))
    plt.close()

def plot_training_history(save_path, losses_train, losses_test, 
                         rmses_train, rmses_test,
                         rmses_train_reconstructed, rmses_test_reconstructed):
    """
    Plot training history curves
    
    Args:
        save_path (str): Directory to save the plots
        losses_train (list): Training losses
        losses_test (list): Testing losses
        rmses_train (list): Training RMSEs
        rmses_test (list): Testing RMSEs
        rmses_train_reconstructed (list): Training reconstructed RMSEs
        rmses_test_reconstructed (list): Testing reconstructed RMSEs
    """
    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(losses_train, label="Train Loss")
    plt.plot(losses_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "losses.png"))
    plt.close()

    # RMSE curves
    plt.figure(figsize=(10, 5))
    plt.plot(rmses_train, label="Train RMSE")
    plt.plot(rmses_test, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Testing RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_path, "rmses.png"))
    plt.close()

    # Reconstructed RMSE curves
    plt.figure(figsize=(10, 5))
    plt.plot(rmses_train_reconstructed, label="Train RMSE Reconstructed")
    plt.plot(rmses_test_reconstructed, label="Test RMSE Reconstructed")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Reconstructed")
    plt.title("Training and Testing RMSE Reconstructed")
    plt.legend()
    plt.savefig(os.path.join(save_path, "rmses_reconstructed.png"))
    plt.close()

def save_results(args, model, test_x_tensor, test_target_tensor, test_dataset_new, max, min,
                losses_train=None, losses_test=None, rmses_train=None, rmses_test=None,
                rmses_train_reconstructed=None, rmses_test_reconstructed=None):
    """Save model results and plots"""
    
    # Create save directory
    if args.ode_encoder_decoder:
        args.model_name = f"{args.model_name}-encoder-decoder"
    if args.use_periodic_activation:
        args.model_name = f"{args.model_name}-periodic-activation"
    save_name = f"{args.model_name}_pcs={args.n_pcs}_window={args.window}" 
    if args.use_loss_weights:
        save_name += "_weighted_loss"
    if args.add_sin_cos:
        save_name += "_sin_cos"
    save_path = f"results/baseline_models/{save_name}"
    os.makedirs(save_path, exist_ok=True)

    # Generate predictions
    if args.model_name in ["arima", "arimax"]:
        # These models return numpy arrays directly
        output = model.predict(test_x_tensor.squeeze(1), args.horizon)
        output_np = batch_data_to_timeseries(output, n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
    elif args.model_name == "garch":
        # GARCH returns both predictions and volatility
        output, volatility = model.predict(test_x_tensor.squeeze(1), args.horizon)
        output_np = batch_data_to_timeseries(output, n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
        volatility_np = batch_data_to_timeseries(volatility, n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
        # Save volatility predictions
        np.save(os.path.join(save_path, "test_volatility.npy"), volatility_np)
    else:
        # Neural models and GP
        with torch.no_grad():
            n_samples = args.n_samples
            if args.model_name == "kalman":
                output, output_cov = model(test_x_tensor.squeeze(1), args.horizon)
                output = output.unsqueeze(1)
            elif args.model_name == "nsde":
                # Generate multiple trajectories for NSDE
                output = model(test_x_tensor.squeeze(1), n_samples=n_samples)
            elif args.model_name == "gp":
                # Get mean and standard deviation from GP (these are numpy arrays)
                mean, std = model.predict(test_x_tensor.squeeze(1), return_std=True)
                # Generate samples from the predictive distribution
                output = []
                for _ in range(n_samples):
                    # Use numpy random for numpy arrays
                    sample = mean + np.random.randn(*mean.shape) * std
                    output.append(sample)
                output = np.stack(output)  # Shape: (n_samples, B, N, H)
            else: # node, gaussian process
                if args.model_name == "nsde":
                    output = model(test_x_tensor.squeeze(1), n_samples=n_samples)
                    if args.add_sin_cos:
                        output = output[:, :, :-2, :]
                else:
                    output = model(test_x_tensor.squeeze(1))
                    if args.add_sin_cos: 
                        output = output[:, :-2, :]
                output = output.unsqueeze(1)
                print(f"test_x_tensor = {test_x_tensor.shape}")
                print(f"output = {output.shape}")
        
        # Process predictions
        output_np = []
        if args.model_name == "nsde":
            output_np = stochastic_batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
            print(f"output_np = {output_np.shape}")
            for i in range(len(output_np)):
                sample_output = output_np[i]  # Already in correct shape
                if args.use_normalization:
                    sample_output = inverse_normalize(sample_output, max, min)
                print(f"sample_output = {sample_output.shape}")
                output_np[i] = sample_output
            output_np = np.stack(output_np)
        else: # node 
            output_np = batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
            print(f"output_np = {output_np.shape}")
            
            if args.use_normalization:
                output_np = inverse_normalize(output_np, max, min)
        print(f"output_np = {output_np.shape}")

    # Process target
    if args.add_sin_cos:
        test_target_tensor = test_target_tensor[:,:, :-2, :]
    test_target_np = batch_data_to_timeseries(test_target_tensor.cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
    if args.use_normalization:
        test_target_np = inverse_normalize(test_target_np, max, min)


    print(f"output_np = {output_np.shape}")
    print(f"test_target_np = {test_target_np.shape}")
    # Save predictions
    np.save(os.path.join(save_path, "test_pred.npy"), output_np)
    np.save(os.path.join(save_path, "test_true.npy"), test_target_np)

    # Save ENSO reconstructions
    if args.model_name in ["nsde", "gp"]:
        # Handle probabilistic models
        enso34_preds = []
        for i in range(n_samples):
            _, enso34_pred = reconstruct_enso(pcs=output_np[i], 
                                            real_pcs=test_target_np,
                                            top_n_pcs=args.n_pcs,
                                            flag="test")
            enso34_preds.append(enso34_pred)
        enso34_preds = np.stack(enso34_preds)
    else:
        # Handle deterministic models
        _, enso34_pred = reconstruct_enso(pcs=output_np,
                                        real_pcs=test_target_np,
                                        top_n_pcs=args.n_pcs,
                                        flag="test")
        
    enso34, _ = reconstruct_enso(pcs=test_target_np, 
                                real_pcs=test_target_np,
                                top_n_pcs=args.n_pcs,
                                flag="test")
    
    # Save ENSO predictions
    if args.model_name in ["nsde", "gp"]:
        np.save(os.path.join(save_path, "test_enso34_pred_samples.npy"), enso34_preds)
        np.save(os.path.join(save_path, "test_enso34_true.npy"), enso34)
        # Create ribbon plot
        plot_prediction_with_uncertainty(save_path, enso34, enso34_preds,
                                      f"ENSO Predictions with Uncertainty ({args.model_name})")
    else:
        np.save(os.path.join(save_path, "test_enso34_pred.npy"), enso34_pred)
        np.save(os.path.join(save_path, "test_enso34_true.npy"), enso34)
        # Create standard plot
        plot_predictions(save_path, enso34, enso34_pred,
                       f"ENSO Predictions ({args.model_name})")
    
    # Save training history if available
    if all(x is not None for x in [losses_train, losses_test, 
                                  rmses_train, rmses_test,
                                  rmses_train_reconstructed, rmses_test_reconstructed]):
        plot_training_history(save_path, losses_train, losses_test,
                            rmses_train, rmses_test,
                            rmses_train_reconstructed, rmses_test_reconstructed)
    

    # plot skills 
    final_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, shuffle=False, drop_last=True)
    model.eval()
    true_npy, pred_npy = [], []
    true_nino, pred_nino = [],[] 
    for encoder_input, label in final_loader:
        if args.model_name == "kalman":
            output, output_cov = model(encoder_input.squeeze(1), args.horizon)
            if args.add_sin_cos:
                output = output[:, :-2, :]
                label = label[:, :, :-2, :].squeeze(1)
        elif args.model_name == "nsde":
            output = model(encoder_input.squeeze(1), n_samples=args.n_samples)
            if args.add_sin_cos:
                output = output[:, :, :-2, :]
                label = label[:, :, :-2, :].squeeze(1)
        else:
            output = model(encoder_input.squeeze(1))
            if args.add_sin_cos:
                if len(output.shape) == 4: # stochastic models
                    output = output[:, :, -2, :]
                else:
                    output = output[:, :-2, :]
                label = label[:, :, :-2, :].squeeze(1)
        if args.model_name == "nsde":
            pred_np = stochastic_batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
            pred_np = np.mean(pred_np, axis=0)
        else:
            pred_np = batch_data_to_timeseries(output.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
        label_np = batch_data_to_timeseries(label.detach().cpu().numpy(), n_pcs=args.n_pcs, sin_cos=args.add_sin_cos)
        if args.use_normalization:
            label_np = inverse_normalize(label_np, max, min)
            pred_np = inverse_normalize(pred_np, max, min)
        nino34, nino34_pred = reconstruct_enso(pcs=pred_np, real_pcs=label_np, top_n_pcs=args.n_pcs, flag="test")
        true_npy.append(np.expand_dims(label_np, 0))
        pred_npy.append(np.expand_dims(pred_np, 0))
        true_nino.append(np.expand_dims(nino34, 0))
        pred_nino.append(np.expand_dims(nino34_pred, 0))
    
    true_npy = np.concatenate(true_npy, axis=0)
    pred_npy = np.concatenate(pred_npy, axis=0)
    true_nino = np.concatenate(true_nino, axis=0)
    pred_nino = np.concatenate(pred_nino, axis=0)
    print(true_npy.shape)
    print(pred_npy.shape)
    print(true_nino.shape)
    print(pred_nino.shape)

    np.save(os.path.join(save_path, f"test_pred_batched.npy"), pred_npy)
    np.save(os.path.join(save_path, f"test_label_batched.npy"), true_npy)
    np.save(os.path.join(save_path, f"test_enso_reconstructed_batched.npy"), pred_nino)
    np.save(os.path.join(save_path, f"test_enso_batched.npy"), true_nino)

    # skills
    plot_channel_rmse(pred_npy, true_npy, args.model_name, n_pcs=args.n_pcs, save_path=save_path)
    plot_ts_channel_rmse(output_np, test_target_np, args.model_name, n_pcs=args.n_pcs, save_path=save_path)
    plot_enso_anomaly_correlation(pred_nino, true_nino, args.model_name, save_path)
    plot_enso_anomaly_rmse(pred_nino, true_nino, args.model_name, save_path)
    plot_enso_forecast_vs_real(pred_nino, true_nino, args.model_name, save_path)    
    



def plot_training_curves(save_path, losses_train, losses_test, rmses_train, rmses_test,
                        rmses_train_reconstructed, rmses_test_reconstructed):
    """Plot and save training curves"""
    
    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(losses_train, label="Train Loss")
    plt.plot(losses_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "losses.png"))
    plt.close()

    # RMSE curves
    plt.figure(figsize=(10, 5))
    plt.plot(rmses_train, label="Train RMSE")
    plt.plot(rmses_test, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Testing RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_path, "rmses.png"))
    plt.close()

    # Reconstructed RMSE curves
    plt.figure(figsize=(10, 5))
    plt.plot(rmses_train_reconstructed, label="Train RMSE Reconstructed")
    plt.plot(rmses_test_reconstructed, label="Test RMSE Reconstructed")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Reconstructed")
    plt.title("Training and Testing RMSE Reconstructed")
    plt.legend()
    plt.savefig(os.path.join(save_path, "rmses_reconstructed.png"))
    plt.close()

def plot_gp_training_loss(save_path, losses_per_dim, model_name="GP"):
    """
    Plot training losses for GP model
    
    Args:
        save_path (str): Directory to save the plot
        losses_per_dim (list): List of lists containing losses for each dimension
        model_name (str): Name of the model for the plot title
    """
    plt.figure(figsize=(10, 5))
    
    for dim, losses in enumerate(losses_per_dim):
        plt.plot(losses, label=f'Dimension {dim}')
    
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title(f"{model_name} Training Loss per Dimension")
    plt.legend()
    plt.savefig(os.path.join(save_path, "gp_training_loss.png"))
    plt.close() 