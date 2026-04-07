import torch
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF, RationalQuadratic
import scipy.optimize
import pandas as pd 
import seaborn as sns 
import numpy as np

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

def seasonal(t, amplitude, period):
    """Generate a sinusoidal curve."""
    y1 = amplitude * np.sin((2*np.pi)*t/period) 
    return y1


def optimizer(obj_func, x0, bounds):
    res = scipy.optimize.minimize(
        obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
        options={'maxiter': 1000}
    )
    return res.x, res.fun


def run_gp_regression(train_y, test_y, epochs):
    train_time = np.linspace(0, 699, 700).reshape(-1, 1)  # Training: 700 time stamps (0 to 699)
    test_time = np.linspace(700, 899, 200).reshape(-1, 1)
    time = np.concatenate((train_time, test_time), axis=0)
    
    # NOTE: DEBUG
    # Add two seasonal components. 
    # y1 = np.apply_along_axis(lambda t : seasonal(t, amplitude=2, period=40), axis=1, arr=time)
    # train_y = y1[:700]
    # test_y = y1[700:]
   
    y1 = np.concatenate((train_y, test_y), axis=0)
    # Initialize the likelihood and the model
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))

    k1 = ConstantKernel(2.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=12, periodicity_bounds=(10, 15))

    k2 = ConstantKernel(2.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=50.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2))

    kernel = k0 + k1 + k2
    # The 'alpha' parameter here is set to the noise variance (0.1**2)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,  # Increase the number of restarts
        alpha=0.1**2,
        optimizer=optimizer
    )

    gp.fit(train_time, train_y)

    predicted_mean, predicted_std = gp.predict(time, return_std=True)

    data_df = pd.DataFrame({'time': time.squeeze(), "y1": y1.squeeze()})
    data_df['y_pred'] = predicted_mean
    data_df['y_std'] = predicted_std
    data_df['y_pred_lwr'] = data_df['y_pred'] - 2*data_df['y_std']
    data_df['y_pred_upr'] = data_df['y_pred'] + 2*data_df['y_std']

    if not os.path.exists("results/gp_regression"):
        os.makedirs("results/gp_regression")

    # Plot the results
    fig, ax = plt.subplots()

    ax.fill_between(
        x=data_df['time'], 
        y1=data_df['y_pred_lwr'], 
        y2=data_df['y_pred_upr'], 
        color=sns_c[2], 
        alpha=0.15, 
        label='credible_interval'
    )

    sns.lineplot(x='time', y='y1', data=data_df, color=sns_c[0], label = 'y1', ax=ax)
    sns.lineplot(x='time', y='y_pred', data=data_df, color=sns_c[2], label='y_pred', ax=ax)

    ax.axvline(700, color=sns_c[3], linestyle='--', label='train-test split')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(title='Prediction Sample 1', xlabel='time', ylabel='ENSO Anomaly');
    plt.savefig("results/gp_regression/gp_regression.png")

    from sklearn.metrics import mean_absolute_error
    print(f'R2 Score Train = {gp.score(X=train_time, y=train_y): 0.3f}')
    print(f'R2 Score Test = {gp.score(X=test_time, y=test_y): 0.3f}')
    print(f'MAE Train = {mean_absolute_error(y_true=train_y, y_pred=gp.predict(train_time)): 0.3f}')
    print(f'MAE Test = {mean_absolute_error(y_true=test_y, y_pred=gp.predict(test_time)): 0.3f}')
