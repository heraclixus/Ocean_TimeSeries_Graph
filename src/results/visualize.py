import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
import os
from scipy import stats
import xarray as xr
import xskillscore as xs


category_1_features = ["nina1", "nina3", "nina34", "nina4"]
category_2_features = category_1_features + ["pacwarm", 'soi', "tni", "whwp"]
category_3_features = category_2_features + ["ammsst", "tna", "tsa", 'amon']
category_4_features = category_3_features + ["ea", "epo", "nao", "pna", "wp"]
category_5_features = category_4_features + ["qbo", "solar"]


# climate anomay score
def calculate_acc(true, pred):
    return xs.pearson_r(xr.DataArray(true), xr.DataArray(pred))

# calculate rmse 
def calcaulate_rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


def plot_forecast_skill(folder_path, true, pred, pred_len=24):

    horizon_sizes = [i for i in range(3, pred_len)]
    accs, rmses = [],[]
    for pred_horizon in horizon_sizes:
        pred_temp = pred[:, pred_horizon].flatten()
        true_temp = true[:, pred_horizon].flatten()
        acc = calculate_acc(true_temp, pred_temp)
        rmse = calcaulate_rmse(true_temp, pred_temp)
        accs.append(acc)
        rmses.append(rmse)
    fig, ax = plt.subplots(2, 1, figsize=(5,10), sharex=True)
    ax[0].plot(horizon_sizes, accs, linestyle='--', marker='o', color='g', label='ACC')
    ax[1].plot(horizon_sizes, rmses, linestyle='--', marker='o', color='b', label='RMSE')
    ax[0].set_xlabel("prediction horizon size")
    ax[0].set_ylabel("Climate Anomaly (ACC)")
    ax[1].set_ylabel("RMSE")
    plt.title(f"{folder_path} Forecast Skills")
    plt.savefig(os.path.join(folder_path, f'{folder_path}_forecast_skill.png'))
    plt.close()


def plot_predictions_vs_true_reg(folder_path, visualize_length=50000, pred_len=24):
    if "cat1" in folder_path:
        features = category_1_features
    elif "cat2" in folder_path:
        features = category_2_features
    elif "cat3" in folder_path:
        features = category_3_features
    elif "cat4" in folder_path:
        features = category_4_features
    else:
        features = category_5_features

    n_features = len(features)
    nino_feature_index = features.index("nina34")

    # Load the data
    test_pred = np.load(os.path.join(folder_path, 'test_pred.npy')).squeeze().reshape(-1, n_features, pred_len)[:, nino_feature_index,:]
    test_true = np.load(os.path.join(folder_path, 'test_true.npy')).squeeze().reshape(-1, n_features, pred_len)[:, nino_feature_index,:]

    plot_forecast_skill(folder_path, test_true, test_pred, pred_len)

    test_pred = test_pred.flatten()
    test_true = test_true.flatten()

    if len(test_pred) > visualize_length:
        test_pred = test_pred[:visualize_length]
        test_true = test_true[:visualize_length]
    
    # print(f"R2: {r2_score(test_true, test_pred)}")
    # print(f"MAPE: {mean_absolute_percentage_error(test_true, test_pred)}")
    # print(f"RMSE: {calcaulate_rmse(test_true, test_pred)}")
    # print(f"MAE: {mean_absolute_error(test_true, test_pred)}")

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(test_true, test_pred, alpha=0.5)
    
    # Add diagonal line for perfect predictions
    min_val = min(test_true.min(), test_pred.min())
    max_val = max(test_true.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Nino3.4 Predictions vs True Values')
    plt.legend()
    
    # Save the plot
    folder_name = os.path.basename(folder_path)
    plt.savefig(os.path.join("../plots", f'{folder_name}_nino34_reg.png'))




def plot_predictions_vs_true(folder_path, visualize_length=500):
    if "cat1" in folder_path:
        features = category_1_features
    elif "cat2" in folder_path:
        features = category_2_features
    elif "cat3" in folder_path:
        features = category_3_features
    elif "cat4" in folder_path:
        features = category_4_features
    else:
        features = category_5_features

    n_features = len(features)  
    nino_feature_index = features.index("nina34")

    # Load the data
    test_pred = np.load(os.path.join(folder_path, 'test_pred.npy')).squeeze().reshape(-1, n_features, 24)[:, nino_feature_index,:]
    test_true = np.load(os.path.join(folder_path, 'test_true.npy')).squeeze().reshape(-1, n_features, 24)[:, nino_feature_index,:]

    plot_forecast_skill(folder_path, test_true, test_pred, 24)

    test_pred = test_pred.flatten()
    test_true = test_true.flatten()
    
    if len(test_pred) > visualize_length:
        test_pred = test_pred[:visualize_length]
        test_true = test_true[:visualize_length]
    
    # Create time index
    time_steps = np.arange(len(test_true))
    
    plt.figure(figsize=(15, 6))
    
    # Plot both series
    plt.plot(time_steps, test_true, label='True Values', alpha=0.7)
    plt.plot(time_steps, test_pred, label='Predicted Values', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Predicted vs True Values Over Time')
    plt.legend()
    
    # Save the plot
    folder_name = os.path.basename(folder_path)
    plt.savefig(os.path.join("../plots", f'{folder_name}_nino34_timeseries.png'))
    plt.close()



# Get all subdirectories in the current directory
subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Process each subdirectory
for subdir in subdirs:
    try:
        plot_predictions_vs_true_reg(subdir)
        plot_predictions_vs_true(subdir)
        print(f"Successfully processed {subdir}")
    except Exception as e:
        print(f"Error processing {subdir}: {str(e)}")