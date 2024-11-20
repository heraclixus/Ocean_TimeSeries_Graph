import os
import numpy as np
import glob
import pandas as pd

categories = []
fourier_coeffs = []
periods = []
mapes = []
rmses = [] 

for file in os.listdir("./"):
    if file.endswith("txt") and "log_cat" in file and "nino_" in file:
        category = file.split("_")[1]
        configs = file.split("_")[-1][:-4]
        # print(f"{category}_{configs}")
        fourier_coeff = int(configs[0])
        period = int(configs[1:])

        best_epoch = 0
        with open(file, "r") as f:
            lines = f.readlines()
            best_epoch = lines[-1].split(" ")[-1][:-2]
            if len(best_epoch) == 1:
                best_epoch = f"Epoch 000{best_epoch}"
            elif len(best_epoch) == 2:
                best_epoch = f"Epoch 00{best_epoch}"
            else:
                best_epoch = f"Epoch 0{best_epoch}"
            best_epoch += " [Test seq"

            isfound = False   
            for line in lines:
                if best_epoch in line:
                    print(line)
                    rmse = float(line.split("|")[2].split(" ")[-2])
                    mape = float(line.split("|")[4].split(" ")[-2])
                    rmses.append(rmse)
                    mapes.append(mape)
                    fourier_coeffs.append(fourier_coeff)
                    periods.append(period)
                    categories.append(category)
                    isfound = True
                    break
            # not found 
            if not isfound: 
                rmses.append(np.nan)
                mapes.append(np.nan)
                fourier_coeffs.append(fourier_coeff)
                periods.append(period)
                categories.append(category)


df = pd.DataFrame({
    "categories": categories,
    "fourier_coeffs": fourier_coeffs,
    "periods": periods,
    "rmses": rmses,
    "mapes": mapes
})

df.to_csv("./summary_training_logs.csv")
  
  