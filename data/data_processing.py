import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_ocean_index_data(data_file):
    with open(data_file, "r") as f:
        data_content = f.readlines()
        data_content = [item.rstrip("\n") for item in data_content]
        data_content = [item.lstrip(" ") for item in data_content]
        data_content = [item for item in data_content if len(item.split(" ")) == 13]
        years = [item.split(" ")[0] for item in data_content]
        ts = np.array([item.split(" ")[1:] for item in data_content]).T
        cols = [i for i in range(1, 13)]
        assert len(ts) == len(cols)
        df_dict = dict(zip(cols, ts))
        df_dict["year"] = years
        df = pd.DataFrame.from_dict(df_dict).set_index("year")
        if "2024" in df.index:
            df = df.drop(index="2024")
            df = df.drop(index="2023")
        filename = data_file.split(".")[0] + ".csv"
        df.to_csv(filename)

nino_years = pd.read_csv("nina3.csv").year.to_list()[:-1]
print(nino_years)
nino_years = [int(item) for item in nino_years]
year_month = []
for year in nino_years:
    for i in range(1,13):
        year_month.append(f"{year}_{i}")

dfs = []
names = []
import os
for file in os.listdir("."):
    if file.endswith(".csv") and file != "ocean_timeseries.csv" and file != "pdo.csv":
        df = pd.read_csv(file)
        year = df.year.to_list()
        year = [int(elem) for elem in year]
        if min(year) <= min(nino_years) and max(year) >= max(nino_years):
            df = df.loc[df.year.isin(nino_years)]
            dfs.append(df)
            names.append(file.split(".")[0])
print(len(dfs))
print(names)


combined_dicts = {}
for i in range(len(dfs)):
    ts = dfs[i].drop(["year"], axis=1).to_numpy().flatten()
    print(f"index = {names[i]}, len = {len(ts)}")
    combined_dicts[names[i]] = ts

df_combined = pd.DataFrame.from_dict(combined_dicts)    
df_combined.index = year_month
df_cleaned = df_combined[~df_combined.isin([-99.99, -9.90]).any(axis=1)]
df_cleaned.to_csv("ocean_timeseries.csv")


import seaborn as sns

columns = df_cleaned.columns[1:]

for col in columns:
    plt.figure(figsize=(8,8))  # Create a new figure for each plot
    sns.histplot(df_cleaned[col], kde=True)  # Plot histogram with KDE
    plt.title(f'Distribution of {col}') 
    plt.savefig(f"{col}.png")
    plt.close()