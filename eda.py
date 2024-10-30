import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = pd.read_csv("data/indices_ocean_19_timeseries.csv", index_col=0)
data['year'] = list(map(lambda idx: int(idx[:4]), data.index))
data['idx_in_year'] = list(map(lambda idx: int(idx[-1]), data.index))
year_info = data[['year', 'idx_in_year']]
print(year_info)
scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(data.drop(['year', 'idx_in_year'], axis=1)), columns=data.columns[:-2], index=data.index)
data = pd.concat([scaled_data, year_info], axis=1)

train_set = data[data["year"] < 2010]
test_set = data[data["year"] >= 2010]

dir = "aux_data"




# if not os.path.exists(dir):
#     os.makedirs(dir)
# train_set.to_csv(os.path.join(dir, "train.csv"))
# test_set.to_csv(os.path.join(dir, "test.csv"))

# matrix = data.corr()
# matrix.style.background_gradient()

# # Create a heatmap for the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.savefig(os.path.join(dir, 'corr_matrix.png'), dpi=300, bbox_inches='tight')


forecast_horizon = 12 
train_window = 24     
targets = ["nina1","nina3","nina34","nina4"]

def create_lagged_features(data, window_size):
    lagged_data = {}
    for col in data.columns.drop(targets):
        for lag in range(1, window_size + 1):
            lagged_data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return pd.DataFrame(lagged_data)

# Apply lagged features to training and test sets
train_lagged = create_lagged_features(train_set, train_window).dropna()
test_lagged = create_lagged_features(test_set, train_window).dropna()

# Define evaluation metics
def evaluate(predictions, targets):
    return {
        'MAE': mean_absolute_error(targets, predictions),
        'RMSE': np.sqrt(mean_squared_error(targets, predictions))
    }


# Align the forecast target
train_targets = train_set[targets].shift(-forecast_horizon).dropna().loc[train_lagged.index[0]:]
train_lagged = train_lagged.iloc[:train_lagged.index.get_loc(train_targets.index[-1])+1]
test_targets = test_set[targets].shift(-forecast_horizon).dropna().loc[test_lagged.index[0]:]
test_lagged = test_lagged.iloc[:test_lagged.index.get_loc(test_targets.index[-1])+1]

models = {}
train_preds = pd.DataFrame(index=train_targets.index, columns=targets)
test_preds = pd.DataFrame(index=test_targets.index, columns=targets)

for col in targets:
    model = LinearRegression()
    model.fit(train_lagged, train_targets[col])
    train_preds[col] = model.predict(train_lagged)
    test_preds[col] = model.predict(test_lagged)
    models[col] = model
# # Calculate metrics for each index
results = {}
for col in test_targets.columns:
    results[col] = evaluate(test_preds[col], test_targets[col])

# Display results
for index, metrics in results.items():
    print(f"{index} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

test_res_dir = "aux_data/tested_nina_pred"
if not os.path.exists(test_res_dir):
    os.makedirs(test_res_dir)

for target in targets:
    plt.figure(figsize=(45, 6))
    plt.plot(test_targets.index, test_targets[target], label='Actual')
    plt.plot(test_targets.index, test_preds[target], label='Predicted', linestyle='--')
    plt.title(f'Forecast vs Actual for {target}')
    plt.legend()
    plt.xticks(ticks=test_targets.index[::5], rotation=45)
    plt.savefig(os.path.join(test_res_dir, f'{target}.png'), dpi=300, bbox_inches='tight')
    # plt.show()




