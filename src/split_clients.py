import os
import pandas as pd
import numpy as np

N_CLIENTS = 5

DATA_X = "data/X_processed.csv"
DATA_Y = "data/y_labels.csv"
OUT_DIR = "data/clients"

print("Loading processed data")
X = pd.read_csv(DATA_X)
y = pd.read_csv(DATA_Y).values.ravel()
X = X.select_dtypes(include=[np.number])
print("Numeric feature shape:", X.shape)
print("Data shape:", X.shape, y.shape)

idx = np.arange(len(X))
np.random.shuffle(idx)

splits = np.array_split(idx, N_CLIENTS)

for i, s in enumerate(splits):
    client_dir = f"{OUT_DIR}/client_{i}"
    os.makedirs(client_dir, exist_ok=True)

    X.iloc[s].to_csv(f"{client_dir}/X.csv", index=False)
    pd.Series(y[s]).to_csv(f"{client_dir}/y.csv", index=False)

    attack_ratio = y[s].mean()
    print(f"Client {i} samples:", len(s), "Attack ratio:", round(attack_ratio, 3))

print("Client split completed")

