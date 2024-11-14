import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import os


def plot_predictions_vs_true_reg(folder_path, visualize_length=50000):
    # Load the data
    test_pred = np.load(os.path.join(folder_path, 'test_pred.npy')).squeeze().flatten()
    test_true = np.load(os.path.join(folder_path, 'test_true.npy')).squeeze().flatten()

    if len(test_pred) > visualize_length:
        test_pred = test_pred[:visualize_length]
        test_true = test_true[:visualize_length]
    
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
    plt.savefig(os.path.join(folder_path, f'{folder_name}_nino34_reg.png'))




def plot_predictions_vs_true(folder_path, visualize_length=500):
    # Load the data
    test_pred = np.load(os.path.join(folder_path, 'test_pred.npy')).squeeze().flatten()
    test_true = np.load(os.path.join(folder_path, 'test_true.npy')).squeeze().flatten()
    
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
    plt.savefig(os.path.join(folder_path, f'{folder_name}_nino34_timeseries.png'))
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
