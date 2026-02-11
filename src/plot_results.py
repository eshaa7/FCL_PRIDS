import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/federated_convergence.csv")

plt.figure(figsize=(6, 4))

plt.plot(
    df["round"],
    df["global_accuracy"],
    label="Global Model"
)

plt.plot(
    df["round"],
    df["mean_client_accuracy"],
    linestyle="--",
    label="Mean Client"
)

plt.fill_between(
    df["round"],
    df["min_client_accuracy"],
    df["max_client_accuracy"],
    alpha=0.25,
    label="Client Variance"
)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/federated_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

