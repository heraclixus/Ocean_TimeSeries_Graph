import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("summary_training_logs.csv")
cats = ["cat1","cat2","cat3","cat4","cat5"]


# box plot for each category
def plot_performance_categories(cat="cat",metric="rmse"):
    plt.figure(figsize=(8,6))
    data.boxplot(column=metric,by=cat, grid=False)
    plt.title(f"{metric} for each {cat}")
    xlabels = "Category" if cat=="cat" else cat
    plt.xlabel(xlabels)
    plt.ylabel(metric)
    plt.savefig(f"plots/plot_box_{cat}_{metric}.png")

for cat in ["categories", "fourier_coeffs", "periods"]:
    for metric in ["rmses", "mapes"]:
        plot_performance_categories(cat, metric)

# zero fourier coeff and default period, plot for each category 
plt.figure(figsize=(8,6))
data = data[(data["fourier_coeffs"]==0) & (data["periods"]==10000)]
plt.plot(data["categories"], data["rmses"], c='blue', 
            linestyle='--', marker='o', alpha=0.8, label='RMSE')
plt.title('Scatter Plot of category vs RMSE', fontsize=16)
plt.xlabel('X Values', fontsize=14)
plt.ylabel('Y Values', fontsize=14)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig("plots/plot_scatter_cat_rmse.png")
plt.close()

# zero fourier coeff and default period, plot for each category 
plt.figure(figsize=(8,6))
data = data[(data["fourier_coeffs"]==0) & (data["periods"]==10000)]
plt.plot(data["categories"], data["mapes"], c='blue', 
              linestyle="--", marker='o', alpha=0.8, label='MAPE')
plt.title('Scatter Plot of category vs MAPE', fontsize=16)
plt.xlabel('X Values', fontsize=14)
plt.ylabel('Y Values', fontsize=14)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig("plots/plot_scatter_cat_mape.png")


