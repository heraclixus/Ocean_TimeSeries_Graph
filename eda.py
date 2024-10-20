import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/indices_ocean_19_timeseries.csv", index_col=0)
data['year'] = list(map(lambda idx: int(idx[:4]), data.index))
data['idx_in_year'] = list(map(lambda idx: int(idx[-1]), data.index))
print(data.head())

# train_set = data[data["year"] < 2010]
# test_set = data[data["year"] >= 2010]

dir = "aux_data"

# if not os.path.exists(dir):
#     os.makedirs(dir)
# train_set.to_csv(os.path.join(dir, "train.csv"))
# test_set.to_csv(os.path.join(dir, "test.csv"))

matrix = data.corr()
matrix.style.background_gradient()

# Create a heatmap for the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.savefig(os.path.join(dir, 'corr_matrix.png'), dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()

import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


X = data[['X1', 'X2', 'X3']]  # Input features (multiple columns)
y = data['y']  # Target output (one column)



# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
