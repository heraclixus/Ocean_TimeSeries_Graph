import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import os
import xarray as xr
import xskillscore as xs
# from utils_pca import reconstruct_enso

# Set the width of each bar and positions for x-axis
bar_width = 0.15
models = ["agcrn", "fgnn", "mtgnn", "stemgnn", "pgode"]
components = np.arange(20)


# prediction = (T, 20)
def plot_ts_channel_rmse(prediction, test, model_name, n_pcs=20, save_path=None):
    rmses = [] 
    prediction = prediction.T 
    test = test.T
    for i in range(len(prediction)):
        rmse_per_channel = np.sqrt(np.mean((prediction[i]-test[i])**2, axis=0))
        rmses.append(rmse_per_channel)
    # Create bar plot
    plt.figure(figsize=(12, 6))
    channels = np.arange(1, n_pcs+1)  # 1-20 for x-axis labels
    plt.bar(channels, rmses)
    # Customize plot
    plt.title('RMSE per PC', fontsize=14)
    plt.xlabel('PC', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(channels)
    # Add value labels on top of each bar
    for i, v in enumerate(rmses):
        plt.text(i + 1, v, f'{v:.3f}', 
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_name}_rmse_perchannel_ts.png"), dpi=300, bbox_inches='tight')    
    plt.close()
    
     

# plots for pred and true values, forecast skills
# prediction = (B, 24, 20)
def plot_channel_rmse(prediction, test, model_name, n_pcs=20, save_path=None):
    rmses =[] 
    for i in range(len(prediction)):
        # Compute RMSE for each channel
        rmse_per_channel = np.expand_dims(np.sqrt(np.mean((prediction[i] - test[i])**2, axis=0)),0) # (1,20)
        rmses.append(rmse_per_channel) 
    rmses = np.mean(np.concatenate(rmses), axis=0)
    # Create bar plot
    plt.figure(figsize=(12, 6))
    channels = np.arange(1, n_pcs+1)  # 1-20 for x-axis labels
    plt.bar(channels, rmses)
    # Customize plot
    plt.title('RMSE per PC', fontsize=14)
    plt.xlabel('PC', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(channels)
    
    # Add value labels on top of each bar
    for i, v in enumerate(rmses.tolist()):
        plt.text(i + 1, v, f'{v:.3f}', 
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{model_name}_rmse_perchannel.png"), dpi=300, bbox_inches='tight')    
    plt.close()


# plot forecast skill
# prediction = (B, 24)
def plot_enso_anomaly_correlation(prediction, test, model_name, save_path=None):
    prediction = prediction.T  # (24, B)
    test = test.T
    anomaly_correlations = []
    
    for i in range(len(prediction)):
        model_forecast = prediction[i]
        model_true = test[i]
        anomaly_correlations.append(xs.pearson_r(xr.DataArray(model_forecast), xr.DataArray(model_true)))
    lead_times = np.arange(len(prediction))
    # x axis is the lead time, y axis is the acc 
    plt.figure(figsize=(10, 5))
    plt.plot(lead_times, anomaly_correlations, marker="o")
    plt.title(f"{model_name}: Anomaly Correlation Coefficient Comparison for ENSO")
    plt.xlabel("Lead Time")
    plt.ylabel("Anomaly Correlation Coefficient")
    plt.savefig(os.path.join(save_path, f"{model_name}_enso_anomaly_correlation_comparison.png"))
    plt.close()


def plot_enso_anomaly_rmse(prediction, test, model_name, save_path=None):
    prediction = prediction.T  # (24, B)
    test = test.T
    anomaly_rmses = []
    for i in range(len(prediction)):
        model_forecast = prediction[i]
        model_true = test[i]
        anomaly_rmses.append(np.sqrt(np.mean((model_forecast-model_true) ** 2)))
    lead_times = np.arange(len(prediction))
    # x axis is the lead time, y axis is the acc 
    plt.figure(figsize=(10, 5))
    plt.plot(lead_times, anomaly_rmses, marker="o")
    plt.title(f"{model_name}: Anomaly RMSE Comparison for ENSO")
    plt.xlabel("Lead Time")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(save_path, f"{model_name}_enso_anomaly_rmse_comparison.png"))
    plt.close()


# prediction = (B, 24)
def plot_enso_forecast_vs_real(prediction, test, model_name, save_path=None):
    leads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for lead in leads:
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(len(test)), test[:,lead-1], marker="o", label=f"Real lead = {lead}")
        plt.plot(np.arange(len(prediction)), prediction[:, lead-1], marker="x", label=f"Forecast lead = {lead}")
        plt.title(f"{model_name}: ENSO Forecast vs Real, lead = {lead}")
        plt.xlabel("Time")
        plt.ylabel("ENSO")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"{model_name}_enso_forecast_vs_real_{lead}.png"))
        plt.close()


# plots based on log directory

def plot_dimensionwise_rmse_for_models(models):
    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Plot bars for each model with offset positions
    for idx, model in enumerate(models):
        # calculate the test rmse for each model
        model_forecast = np.load(f"{model}_best/test_pred.npy")
        model_true = np.load(f"{model}_best/test_true.npy")
        # component wise rmse 
        test_rmses = np.sqrt(np.mean((model_forecast - model_true) ** 2, axis=0))
        
        # Calculate x positions for this model's bars
        x_positions = components + (idx - len(models)/2 + 0.5) * bar_width
        
        # Plot bars
        bars = plt.bar(x_positions, test_rmses, bar_width, label=model)
        
        # Add value labels on top of bars
        for i, v in enumerate(test_rmses):
            plt.text(x_positions[i], v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Customize the plot
    plt.title('Test RMSE Comparison Across Models', fontsize=14)
    plt.xlabel('Component', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.legend(fontsize=10)

    # Set x-axis ticks to be centered for each component group
    plt.xticks(components, components)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig("test_rmse_comparison.png")



# now load enso data for each model and plot the forecast and true
# for each model, store a list of rmses for different lead times from 1 to 24
def plot_rmse_and_anomaly_correlation(models, pc=1):
    rmses = {}
    anomaly_correlations = {}
    for model in models:
        rmses[model] = []
        anomaly_correlations[model] = []
        model_forecast = np.load(f"{model}_best/test_pred.npy")[pc-1]
        model_true = np.load(f"{model}_best/test_true.npy")[pc-1]        
        for lead_time in range(1, 25):
            rmses[model].append(np.sqrt(np.mean((model_forecast[:lead_time] - model_true[:lead_time]) ** 2)))
            anomaly_correlations[model].append(xs.pearson_r(xr.DataArray(model_forecast[:lead_time]), xr.DataArray(model_true[:lead_time])))

    # plot the acc for each model 
    # x axis is the lead time, y axis is the acc 
    # each model has a line
    plt.figure(figsize=(10, 5))
    for model in models:
        plt.plot(anomaly_correlations[model], label=model, marker="o")
    plt.title(f"Anomaly Correlation Coefficient Comparison Across Models for PC {pc}")
    plt.xlabel("Lead Time")
    plt.ylabel("Anomaly Correlation Coefficient")
    plt.legend()
    plt.savefig(f"anomaly_correlation_comparison_pc{pc}.png")

    plt.close()
    plt.figure(figsize=(10, 5))
    for model in models:
        plt.plot(rmses[model], label=model, marker="o")
    plt.title(f"RMSE Comparison Across Models for PC {pc}")
    plt.xlabel("Lead Time")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"rmses_comparison_pc{pc}.png")



# now load enso data for each model and plot the forecast and true
# for each model, store a list of rmses for different lead times from 1 to 24
# def plot_enso_anomaly_correlation(models):
#     anomaly_correlations = {}
#     for model in models:
#         anomaly_correlations[model] = []
#         model_forecast = np.load(f"{model}_best/test_pred.npy")
#         model_true = np.load(f"{model}_best/test_true.npy")
#         model_forecast, model_true = reconstruct_enso(model_forecast, model_true)
#         for lead_time in range(1, 25):
#             anomaly_correlations[model].append(xs.pearson_r(xr.DataArray(model_forecast[:lead_time]), xr.DataArray(model_true[:lead_time])))

#     # plot the acc for each model 
#     # x axis is the lead time, y axis is the acc 
#     # each model has a line
#     plt.figure(figsize=(10, 5))
#     for model in models:
#         plt.plot(anomaly_correlations[model], label=model, marker="o")
#     plt.title(f"Anomaly Correlation Coefficient Comparison Across Models for ENSO")
#     plt.xlabel("Lead Time")
#     plt.ylabel("Anomaly Correlation Coefficient")
#     plt.legend()
#     plt.savefig(f"enso_anomaly_correlation_comparison.png")


# grid data visualization
# original path = (time, lat, lon)
# pred path = (time, lat, lon)
def create_comparison_animation(original_data, 
                                pred_data, 
                                savepath, 
                                output_path='sst_pred_comparison.mp4', 
                                fps=10):
    """
    Create a side-by-side animation of grids and their time series
    """
    # Calculate spatial means for time series
    original_means = original_data.mean(axis=(1,2))
    pred_means = pred_data.mean(axis=(1,2))
    
    # Create figure with subplots: 2x2 grid
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # Grid visualization subplots
    ax1 = fig.add_subplot(gs[0, 0])  # top left
    ax2 = fig.add_subplot(gs[0, 1])  # top right
    
    # Time series subplots
    ax3 = fig.add_subplot(gs[1, 0])  # bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # bottom right
    
    # Calculate global min and max for consistent colormap
    vmin = min(original_data.min(), pred_data.min())
    vmax = max(original_data.max(), pred_data.max())
    
    # Initialize grid plots
    im1 = ax1.imshow(original_data[0], 
                     cmap='RdBu_r',
                     aspect='auto',
                     vmin=vmin,
                     vmax=vmax)
    im2 = ax2.imshow(pred_data[0], 
                     cmap='RdBu_r',
                     aspect='auto',
                     vmin=vmin,
                     vmax=vmax)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Temperature Anomaly (°C)')
    plt.colorbar(im2, ax=ax2, label='Temperature Anomaly (°C)')
    
    # Set titles
    ax1.set_title(f'Original Grid ({original_data.shape[1]}x{original_data.shape[2]})')
    ax2.set_title(f'Pred Grid ({pred_data.shape[1]}x{pred_data.shape[2]})')
    
    # Initialize time series plots
    time_points = np.arange(len(original_means))
    line1, = ax3.plot(time_points[0:1], original_means[0:1], 'b-')
    line2, = ax4.plot(time_points[0:1], pred_means[0:1], 'r-')
    
    # Set time series plot properties
    for ax, title in [(ax3, 'Original Grid Mean'), (ax4, 'Pred Grid Mean')]:
        ax.set_xlim(0, len(time_points))
        ax.set_ylim(min(original_means.min(), pred_means.min()),
                   max(original_means.max(), pred_means.max()))
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Spatial Mean Temperature')
        ax.grid(True)
    
    # Add time step counter
    time_text = fig.text(0.5, 0.95, '', ha='center')
    
    def update(frame):
        """Update function for animation"""
        # Update grid plots
        im1.set_array(original_data[frame])
        im2.set_array(pred_data[frame])
        
        # Update time series (show up to current frame)
        line1.set_data(time_points[:frame+1], original_means[:frame+1])
        line2.set_data(time_points[:frame+1], pred_means[:frame+1])
        
        # Update time counter
        time_text.set_text(f'Time step: {frame}')
        
        return im1, im2, line1, line2, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, 
                                 update, 
                                 frames=len(original_data),
                                 interval=1000/fps,
                                 blit=True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save animation
    anim.save(f"{savepath}/{output_path}", writer='ffmpeg', fps=fps)
    plt.close()




def create_comparison_animation_data(original_data, coarse_data, output_path='sst_comparison.mp4', fps=10):
    """
    Create a side-by-side animation of grids and their time series
    """
    # Calculate spatial means for time series
    original_means = original_data.mean(axis=(1,2))
    coarse_means = coarse_data.mean(axis=(1,2))
    
    # Create figure with subplots: 2x2 grid
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # Grid visualization subplots
    ax1 = fig.add_subplot(gs[0, 0])  # top left
    ax2 = fig.add_subplot(gs[0, 1])  # top right
    
    # Time series subplots
    ax3 = fig.add_subplot(gs[1, 0])  # bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # bottom right
    
    # Calculate global min and max for consistent colormap
    vmin = min(original_data.min(), coarse_data.min())
    vmax = max(original_data.max(), coarse_data.max())
    
    # Initialize grid plots
    im1 = ax1.imshow(original_data[0], 
                     cmap='RdBu_r',
                     aspect='auto',
                     vmin=vmin,
                     vmax=vmax)
    im2 = ax2.imshow(coarse_data[0], 
                     cmap='RdBu_r',
                     aspect='auto',
                     vmin=vmin,
                     vmax=vmax)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Temperature Anomaly (°C)')
    plt.colorbar(im2, ax=ax2, label='Temperature Anomaly (°C)')
    
    # Set titles
    ax1.set_title(f'Original Grid ({original_data.shape[1]}x{original_data.shape[2]})')
    ax2.set_title(f'Coarse Grid ({coarse_data.shape[1]}x{coarse_data.shape[2]})')
    
    # Initialize time series plots
    time_points = np.arange(len(original_means))
    line1, = ax3.plot(time_points[0:1], original_means[0:1], 'b-')
    line2, = ax4.plot(time_points[0:1], coarse_means[0:1], 'r-')
    
    # Set time series plot properties
    for ax, title in [(ax3, 'Original Grid Mean'), (ax4, 'Coarse Grid Mean')]:
        ax.set_xlim(0, len(time_points))
        ax.set_ylim(min(original_means.min(), coarse_means.min()),
                   max(original_means.max(), coarse_means.max()))
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Spatial Mean Temperature')
        ax.grid(True)
    
    # Add time step counter
    time_text = fig.text(0.5, 0.95, '', ha='center')
    
    def update(frame):
        """Update function for animation"""
        # Update grid plots
        im1.set_array(original_data[frame])
        im2.set_array(coarse_data[frame])
        
        # Update time series (show up to current frame)
        line1.set_data(time_points[:frame+1], original_means[:frame+1])
        line2.set_data(time_points[:frame+1], coarse_means[:frame+1])
        
        # Update time counter
        time_text.set_text(f'Time step: {frame}')
        
        return im1, im2, line1, line2, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, 
                                 update, 
                                 frames=len(original_data),
                                 interval=1000/fps,
                                 blit=True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()



if __name__ == "__main__":
    original_data = np.random.randn(100, 128, 128)
    coarse_data = np.random.randn(100, 32, 32)
    print(original_data.shape)
    print(coarse_data.shape)
    # create_comparison_animation_data(original_data, coarse_data)