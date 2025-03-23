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
    plot_enso_forecast_vs_real,
    create_comparison_animation
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
                rmses_train_reconstructed=None, rmses_test_reconstructed=None, edge_index=None,
                lat=None, lon=None, indsst=None):
    """Save model results and plots"""
    if args.use_convnet:
        args.model_name = f"{args.model_name}-convnet"
    if args.ode_encoder_decoder:
        args.model_name = f"{args.model_name}-encoder-decoder"
    if args.use_periodic_activation:
        args.model_name = f"{args.model_name}-periodic-activation"
    if args.input_file == "../data/nino34_mat.mat":
        save_name = f"grid_{args.model_name}_window={args.window}"
    else:
        save_name = f"{args.model_name}_pcs={args.n_pcs}_window={args.window}" 
    if args.use_loss_weights:
        save_name += "_weighted_loss"
    if args.add_sin_cos:
        save_name += "_sin_cos"
    if args.use_region_data:
        save_name += "_region_only"
    save_path = f"results/baseline_models/{save_name}"
    os.makedirs(save_path, exist_ok=True)

    # Generate predictions
    if args.model_name in ["arima", "arimax"]:
        # These models return numpy arrays directly
        output_np = model.predict(test_x_tensor.squeeze(1), args.horizon)
        output_np_ts = batch_data_to_timeseries(output_np)
        if args.use_normalization:
            output_np_ts = inverse_normalize(output_np_ts, max, min)
    elif args.model_name == "garch":
        # GARCH returns both predictions and volatility
        output, volatility = model.predict(test_x_tensor.squeeze(1), args.horizon)
        output_np_ts = batch_data_to_timeseries(output)
        volatility_np_ts = batch_data_to_timeseries(volatility)
        if args.use_normalization:
            volatility_np_ts = inverse_normalize(volatility_np_ts, max, min)
            output_np_ts = inverse_normalize(output_np_ts, max, min)
        # Save volatility predictions
        np.save(os.path.join(save_path, "test_volatility.npy"), volatility_np_ts)
    elif args.model_name == "dmd":
        output_np = model.predict(test_x_tensor.squeeze(1), args.horizon)
        output_np_ts = batch_data_to_timeseries(output_np)
        if args.use_normalization:
            output_np_ts = inverse_normalize(output_np_ts, max, min)
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
                print(f"test_x_tensor.shape = {test_x_tensor.shape}")
                if args.model_name == "graphode":
                    output = model(test_x_tensor.squeeze(1), edge_index=edge_index)
                elif args.use_convnet:
                    output = model(test_x_tensor.view(-1, 1, lat, lon, args.window))
                elif args.model_name == "nsde":
                    output = model(test_x_tensor.squeeze(1), n_samples=n_samples)
                    if args.add_sin_cos:
                        output = output[:, :, :-2, :]
                else:
                    output = model(test_x_tensor.squeeze(1))
                    if args.add_sin_cos: 
                        output = output[:, :-2, :]
                if not args.use_convnet:
                    output = output.unsqueeze(1)        
        # Process predictions
        if args.model_name == "nsde" or args.model_name == "gp":
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = output
            output_np_ts = stochastic_batch_data_to_timeseries(output_np)
            for i in range(len(output_np_ts)):
                sample_output = output_np_ts[i]  # Already in correct shape
                if args.use_normalization:
                    sample_output = inverse_normalize(sample_output, max, min)
                output_np_ts[i] = sample_output
            output_np_ts = np.stack(output_np_ts)
        else: 
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = output
            output_np_ts = batch_data_to_timeseries(output_np)
            
            if args.use_normalization:
                output_np_ts = inverse_normalize(output_np_ts, max, min)

    if isinstance(output_np_ts, torch.Tensor):
        output_np_ts = output_np_ts.cpu().numpy()
    # Process target
    if args.add_sin_cos:
        test_target_tensor = test_target_tensor[:,:, :-2, :]
    test_target_np = test_target_tensor.cpu().numpy().squeeze(1)
    test_target_np_ts = batch_data_to_timeseries(test_target_np)
    if args.use_normalization:
        test_target_np_ts = inverse_normalize(test_target_np_ts, max, min)


    print(f"output_np_ts.shape = {output_np_ts.shape}")
    print(f"test_target_np_ts.shape = {test_target_np_ts.shape}")

    # Save predictions
    np.save(os.path.join(save_path, "test_pred.npy"), output_np_ts)
    np.save(os.path.join(save_path, "test_true.npy"), test_target_np_ts)


    # for ENSO region 
    if not args.use_region_data: # only compute index region metrics          
        indsst_tensor = indsst
        output_np_ts = output_np_ts[:,indsst_tensor]
        test_target_np_ts = test_target_np_ts[:,indsst_tensor]

    # Save ENSO reconstructions
    if args.model_name in ["nsde", "gp"]:
        # Handle probabilistic models
        enso34_preds = []
        for i in range(n_samples):
            _, enso34_pred = reconstruct_enso(pcs=output_np_ts[i], 
                                            real_pcs=test_target_np_ts,
                                            top_n_pcs=args.n_pcs,
                                            flag="test")
            enso34_preds.append(enso34_pred)
        enso34_preds = np.stack(enso34_preds)
    else:
        # Handle deterministic models
        if args.input_file == "../data/nino34_mat.mat":
            enso34_pred = output_np_ts.mean(axis=1)
        else:
            _, enso34_pred = reconstruct_enso(pcs=output_np_ts,
                                            real_pcs=test_target_np_ts,
                                            top_n_pcs=args.n_pcs,
                                            flag="test")
    if args.input_file == "../data/nino34_mat.mat":
        enso34 = test_target_np_ts.mean(axis=1)
    else:
        enso34, _ = reconstruct_enso(pcs=test_target_np_ts, 
                                    real_pcs=test_target_np_ts,
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
    if args.model_name in ['nsde', 'node', 'graphode']:

        final_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=1, shuffle=False, drop_last=True)
        if isinstance(model, nn.Module): # only for neural models
            model.eval()
        true_npy, pred_npy = [], []
        true_nino, pred_nino = [],[] 
        true_npy_orig, pred_npy_orig = [], [] 
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
            elif args.model_name == "graphode":
                output = model(encoder_input.squeeze(1), edge_index=edge_index)
                if args.add_sin_cos:
                    output = output[:, :-2, :]
                    label = label[:, :, :-2, :].squeeze(1)
            else:
                if args.use_convnet:
                    output = model(encoder_input.view(-1, 1, lat, lon, args.window))
                else:
                    output = model(encoder_input.squeeze(1))
                if args.add_sin_cos:
                    if len(output.shape) == 4: # stochastic models
                        output = output[:, :, -2, :]
                    else:
                        output = output[:, :-2, :]
                    label = label[:, :, :-2, :].squeeze(1)
            if args.model_name == "nsde":
                pred_np = stochastic_batch_data_to_timeseries(output.detach().cpu().numpy())
                pred_np = np.mean(pred_np, axis=0)
            else:
                pred_np = batch_data_to_timeseries(output.detach().cpu().numpy())
            label_np = batch_data_to_timeseries(label.detach().cpu().numpy())
            if args.use_normalization:
                label_np = inverse_normalize(label_np, max, min)
                pred_np = inverse_normalize(pred_np, max, min)
            pred_npy_orig.append(np.expand_dims(pred_np, 0))
            true_npy_orig.append(np.expand_dims(label_np, 0))
            if not args.use_region_data: # only compute index region metrics          
                indsst_tensor = indsst
                pred_np = pred_np[:,indsst_tensor]
                label_np = label_np[:,indsst_tensor]
            if args.input_file == "../data/nino34_mat.mat":
                nino34, nino34_pred = pred_np.mean(axis=1), label_np.mean(axis=1)
            else:
                nino34, nino34_pred = reconstruct_enso(pcs=pred_np, real_pcs=label_np, 
                                                        top_n_pcs=args.n_pcs, flag="test")
            true_npy.append(np.expand_dims(label_np, 0))
            pred_npy.append(np.expand_dims(pred_np, 0))
            true_nino.append(np.expand_dims(nino34, 0))
            pred_nino.append(np.expand_dims(nino34_pred, 0))
        
        true_npy = np.concatenate(true_npy, axis=0)
        pred_npy = np.concatenate(pred_npy, axis=0)
        true_nino = np.concatenate(true_nino, axis=0)
        pred_nino = np.concatenate(pred_nino, axis=0)
        pred_npy_orig = np.concatenate(pred_npy_orig, axis=0)
        true_npy_orig = np.concatenate(true_npy_orig, axis=0)

    else: # only for non-neural models
        test_recon_rmses = []
        true_npy, pred_npy = [], []
        true_npy_orig, pred_npy_orig = [], [] 
        true_nino, pred_nino = [],[] 
        if args.model_name == "gp":
            output_np = output_np.transpose(1,0,2,3)
        for i in range(len(output_np)):
            if args.model_name == "gp":
                output_np_i = np.expand_dims(output_np[i], 1)
            else:
                output_np_i = np.expand_dims(output_np[i], 0)
            test_target_np_i = np.expand_dims(test_target_np[i], 0)
            if args.model_name == "gp":
                pred_np = stochastic_batch_data_to_timeseries(output_np_i)
                pred_np = np.mean(pred_np, axis=0)
            else:   
                pred_np = batch_data_to_timeseries(output_np_i)
            label_np = batch_data_to_timeseries(test_target_np_i)
            if args.use_normalization:
                label_np = inverse_normalize(label_np, max, min)
                pred_np = inverse_normalize(pred_np, max, min)
            pred_npy_orig.append(np.expand_dims(pred_np, 0))
            true_npy_orig.append(np.expand_dims(label_np, 0))
            if not args.use_region_data: # only compute index region metrics    
                indsst_tensor = indsst
                pred_np = pred_np[:,indsst_tensor]
                label_np = label_np[:,indsst_tensor]
            if args.input_file == "../data/nino34_mat.mat":
                nino34, nino34_pred = pred_np.mean(axis=1), label_np.mean(axis=1)
            else:
                nino34, nino34_pred = reconstruct_enso(pcs=pred_np, real_pcs=label_np, 
                                                        top_n_pcs=args.n_pcs, flag="test")
            test_recon_rmses.append(np.sqrt(np.mean((nino34_pred-nino34)**2)))
            pred_npy.append(np.expand_dims(pred_np, 0))
            true_npy.append(np.expand_dims(label_np, 0))
            pred_nino.append(np.expand_dims(nino34_pred, 0))
            true_nino.append(np.expand_dims(nino34, 0))
        pred_npy = np.concatenate(pred_npy, axis=0)
        true_npy = np.concatenate(true_npy, axis=0)
        pred_nino = np.concatenate(pred_nino, axis=0)
        true_nino = np.concatenate(true_nino, axis=0)
        pred_npy_orig = np.concatenate(pred_npy_orig, axis=0)
        true_npy_orig = np.concatenate(true_npy_orig, axis=0)
        print(f"average test_recon_rmses: {np.mean(np.array(test_recon_rmses))}")

        if args.model_name == "gp":
            output_np_ts = np.mean(output_np_ts, axis=0)

    print(f"pred_npy.shape: {pred_npy.shape}")
    print(f"true_npy.shape: {true_npy.shape}")
    print(f"pred_nino.shape: {pred_nino.shape}")
    print(f"true_nino.shape: {true_nino.shape}")
    print(f"pred_npy_orig.shape: {pred_npy_orig.shape}")
    print(f"true_npy_orig.shape: {true_npy_orig.shape}")

    np.save(os.path.join(save_path, f"test_pred_batched.npy"), pred_npy)
    np.save(os.path.join(save_path, f"test_label_batched.npy"), true_npy)
    np.save(os.path.join(save_path, f"test_enso_reconstructed_batched.npy"), pred_nino)
    np.save(os.path.join(save_path, f"test_enso_batched.npy"), true_nino)
    np.save(os.path.join(save_path, f"test_pred_batched_orig.npy"), pred_npy_orig)
    np.save(os.path.join(save_path, f"test_label_batched_orig.npy"), true_npy_orig)

    if args.input_file == "../data/nino34_mat.mat":
        true_npy_ts = batch_data_to_timeseries(true_npy.transpose(1,2,0)).reshape(-1, lat, lon)
        pred_npy_ts = batch_data_to_timeseries(pred_npy.transpose(1,2,0)).reshape(-1, lat, lon)
        print(f"true_npy_ts.shape: {true_npy_ts.shape}")
        print(f"pred_npy_ts.shape: {pred_npy_ts.shape}")
        np.save(os.path.join(save_path, f"true_npy_ts.npy"), true_npy_ts)
        np.save(os.path.join(save_path, f"pred_npy_ts.npy"), pred_npy_ts)
        # create_comparison_animation(true_npy_ts.reshape(-1, lat, lon), 
        #                             pred_npy_ts.reshape(-1, lat, lon), save_path)
    else:
        plot_channel_rmse(pred_npy, true_npy, args.model_name, n_pcs=args.n_pcs, save_path=save_path)
        plot_ts_channel_rmse(output_np_ts, test_target_np_ts, args.model_name, n_pcs=args.n_pcs, save_path=save_path)
    
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