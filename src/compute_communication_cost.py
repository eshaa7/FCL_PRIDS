import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLIENT_DIR = os.path.join(DATA_DIR, "clients")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

X_sample = pd.read_csv(os.path.join(DATA_DIR, "X_processed.csv"), nrows=1)
num_features = X_sample.select_dtypes(include=[np.number]).shape[1]

num_clients = len(os.listdir(CLIENT_DIR))
rounds = 5
bytes_per_param = 8

params = num_features + 1
model_size_bytes = params * bytes_per_param

federated_comm = rounds * num_clients * model_size_bytes * 2
centralized_comm = 0

df = pd.DataFrame({
    "Method": [
        "Centralized Static",
        "Centralized Iterative",
        "Federated Learning"
    ],
    "Rounds": [
        1,
        rounds,
        rounds
    ],
    "Clients": [
        1,
        1,
        num_clients
    ],
    "Parameters": [
        params,
        params,
        params
    ],
    "Total Communication (MB)": [
        centralized_comm / 1e6,
        centralized_comm / 1e6,
        federated_comm / 1e6
    ]
})

df.to_csv(os.path.join(RESULTS_DIR, "communication_cost.csv"), index=False)

print("Communication cost analysis completed")
print(df)

