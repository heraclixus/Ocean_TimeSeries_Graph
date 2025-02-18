import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

dir_top20 = "logs_report"
dir_top5 = "logs_report_5"

windows = [4,5,6]

graphmodels = ["agcrn", "stemgnn", "mtgnn", "fgnn"]

def obtain_dataframe(dir=dir_top20, top=20):
    df = pd.DataFrame()
    test_rmses_reconstructed = []
    models = []
    batch_sizes = []
    window_sizes = []
    schedules = []
    add_sin_cos = [] 
    for file in os.listdir(dir):
        # if file in problems:
        #     continue
        if file.endswith("txt"):
            try:
                model = file.split("_")[1]
                if model not in graphmodels:
                    continue
                batch_size = file.split("_")[3]
                window_size = file.split("_")[4]
                if len(file.split("_")) == 5:
                    window_size = window_size.split(".")[0]

                with open(os.path.join(dir,file), "r") as f:
                    lines = f.readlines()
                line_with_best_recon = lines[-2]
                # print(line_with_best_recon)
                # print(line_with_test_loss)
                test_rmse = float(line_with_best_recon.split(" ")[-1])
            except:
                print(os.path.join(dir, file))
                continue

        if int(window_size) not in windows:
            continue

        test_rmses_reconstructed.append(test_rmse)
        models.append(model)
        batch_sizes.append(batch_size)
        window_sizes.append(window_size)

        if len(file.split("_")) == 5:
            schedules.append("standard")
            window_size = window_size.split(".")[0]
        elif len(file.split("_")) == 6:
            if file.split("_")[5] == "cosine":
                schedules.append("cosine")
            else:
                schedules.append("standard")
        
        elif len(file.split("_")) == 7:
            if file.split("_")[6] == "warmup":
                schedules.append("cosine_warmup")
            else:
                schedules.append("cosine")
        else: # size is 8 
            schedules.append("cosine_warmup")
        
        if file.split("_")[-1].split(".")[0] != "sin-cos":
            add_sin_cos.append(0)
        else:
            add_sin_cos.append(1)

    df["test_rmse_reconstructed"] = test_rmses_reconstructed
    df["n_pcs"] = [top] * len(df)
    df["model"] = models
    df["batch_size"] = batch_sizes
    df["window_size"] = window_sizes
    df["schedule"] = schedules
    df["add_sin_cos"] = add_sin_cos
    return df

df_20 = obtain_dataframe(dir_top20, top=20)
df_5 = obtain_dataframe(dir_top5, top=5)

df = pd.concat([df_20, df_5])
print(df.shape)
df.to_csv("df_hyperparams_700200.csv", index=False)
