import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

fed = pd.read_csv(os.path.join(RESULTS_DIR, "federated_convergence.csv"))
cent = pd.read_csv(os.path.join(RESULTS_DIR, "centralized_iterative.csv"))

rounds = fed["round"]

global_acc = fed["global_accuracy"]
mean_client = fed["mean_client_accuracy"]

if "client_std" in fed.columns:
    client_std = fed["client_std"]
else:
    client_std = np.abs(global_acc - mean_client)

cent_acc = cent["centralized_accuracy"]

plt.figure(figsize=(6, 4))

plt.plot(
    rounds,
    global_acc,
    marker="o",
    linewidth=2,
    label="Federated Global Model"
)

plt.plot(
    rounds,
    mean_client,
    linestyle="--",
    marker="s",
    linewidth=2,
    label="Federated Mean Client"
)

plt.fill_between(
    rounds,
    global_acc - client_std,
    global_acc + client_std,
    alpha=0.2,
    label="Client Accuracy Variance"
)

plt.plot(
    rounds,
    cent_acc,
    linestyle=":",
    marker="^",
    linewidth=2,
    label="Centralized Iterative"
)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "federated_vs_centralized_convergence.png"),
    dpi=300
)

print("IEEE convergence figure saved")

