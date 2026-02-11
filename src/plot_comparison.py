import pandas as pd
import matplotlib.pyplot as plt

fed = pd.read_csv("results/federated_convergence.csv")
cen = pd.read_csv("results/centralized_convergence.csv")

plt.figure(figsize=(6, 4))

plt.plot(
    fed["round"],
    fed["global_accuracy"],
    marker="o",
    label="Federated Global Model"
)

plt.plot(
    fed["round"],
    fed["mean_client_accuracy"],
    linestyle="--",
    marker="s",
    label="Federated Mean Client"
)

plt.plot(
    cen["round"],
    cen["centralized_accuracy"],
    linestyle=":",
    marker="^",
    label="Centralized Iterative"
)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/centralized_vs_federated.png",
    dpi=300,
    bbox_inches="tight"
)

