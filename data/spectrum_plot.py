import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import pandas as pd
# Simulated multivariate time series (replace with your data)
"""
['pna', 'pacwarmpool', 'pacwarm', 'qbo', 'amon', 'ea', 
'nina1', 'wp', 'nao', 'soi', 'ammsst', 'nina3', 'tsa', 
'nina4', 'gmsst', 'tna', 'whwp', 'epo', 'solar', 'nina34', 
'tni', 'noi']
"""
category_1_features = ["nina1", "nina3", "nina34", "nina4"]
category_2_features = category_1_features + ["pacwarm", 'soi', "tni", "whwp"]
category_3_features = category_2_features + ["ammsst", "tna", "tsa", 'amon']
category_4_features = category_3_features + ["ea", "epo", "nao", "pna", "wp"]
category_5_features = category_4_features + ["qbo", "solar"]

 
np.random.seed(42)
time_series_data = pd.read_csv("ocean_timeseries.csv")[category_5_features].drop(["solar"], axis=1)

# normalize each columns of the time seires
# Min-max scale each column using (x - min)/(max - min)
normalized_data = (time_series_data - time_series_data.min()) / (time_series_data.max() - time_series_data.min())
time_series_data = normalized_data

T, n_series = normalized_data.shape

index_names = time_series_data.columns

assert len(index_names) == n_series

# Frequency and Periodogram Calculation
fig, axes = plt.subplots(n_series//3, 3, figsize=(18,9), sharex=True)
for i in range(n_series//3):
    index = i * 3 
    print(index)
    f, Pxx = periodogram(time_series_data.iloc[:, index], fs=1.0)  # fs is the sampling frequency
    index_name = index_names[index]
    axes[i,0].plot(f, Pxx, label=f'{index_name}', linewidth=1.5)
    axes[i,0].set_ylabel("Power")
    axes[i,0].legend(loc='upper right')
    axes[i,0].grid(True)

    f, Pxx = periodogram(time_series_data.iloc[:, index+1], fs=1.0)
    index_name = index_names[index+1]
    axes[i,1].plot(f, Pxx, label=f'{index_name}', linewidth=1.5)
    axes[i,1].set_ylabel("Power")
    axes[i,1].legend(loc='upper right')
    axes[i,1].grid(True)

    f,Pxx = periodogram(time_series_data.iloc[:, index+2], fs=1.0)
    index_name = index_names[index+2]
    axes[i,2].plot(f, Pxx, label=f'{index_name}', linewidth=1.5)
    axes[i,2].set_ylabel("Power")
    axes[i,2].legend(loc='upper right')
    axes[i,2].grid(True)
    

axes[-1,0].set_xlabel("Frequency")
axes[-1,1].set_xlabel("Frequnecy")
axes[-1,2].set_xlabel("Frequency")
plt.suptitle("Periodograms of Multivariate Ocean indices", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ocean_periodograms.png")