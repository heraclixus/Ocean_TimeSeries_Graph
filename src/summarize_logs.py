import os
import numpy as np
import glob
import pandas as pd

categories = []
fourier_coeffs = []
periods = []
mapes = []
models = []
rmses = [] 

for file in os.listdir("./"):
    if file.endswith("txt") and "log_" in file:
        model = "_".join(file.split("_")[1:])
        
        best_epoch = 0
        with open(file, "r") as f:
            print(file)
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
                    if "pgode" in file:
                        rmse = float(line.split("|")[-3].split(" ")[-2])
                        mape = float(line.split("|")[-2].split(" ")[-2])
                    else:
                        rmse = float(line.split("|")[2].split(" ")[-2])
                        mape = float(line.split("|")[3].split(" ")[-2])
                    rmses.append(rmse)
                    mapes.append(mape)
                    models.append(model)
                    isfound = True
                    break
            # not found 
            if not isfound: 
                rmses.append(np.nan)
                mapes.append(np.nan)
                models.append(model)

df = pd.DataFrame({
    "models": models,
    "rmses": rmses,
    "mapes": mapes
})

df.to_csv("./summary_sst_training_logs.csv")
  
