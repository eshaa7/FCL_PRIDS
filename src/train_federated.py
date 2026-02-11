import os
import copy
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------------- paths and config ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DIR = os.path.join(BASE_DIR, "data", "clients")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ROUNDS = 5
LOCAL_EPOCHS = 1

# ---------------- load global dataset ----------------
print("Loading global test set")

X_full = pd.read_csv(os.path.join(BASE_DIR, "data", "X_processed.csv"))
y_full = pd.read_csv(os.path.join(BASE_DIR, "data", "y_labels.csv")).values.ravel()

X_full = X_full.select_dtypes(include=[np.number]).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    stratify=y_full,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- class weights ----------------
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# ---------------- global model ----------------
global_model = SGDClassifier(
    loss="log_loss",
    max_iter=1,
    learning_rate="optimal",
    tol=None,
    class_weight=class_weights
)

global_model.partial_fit(X_train[:1000], y_train[:1000], classes=classes)

# ---------------- tracking ----------------
client_ids = sorted(os.listdir(CLIENT_DIR))
num_clients = len(client_ids)

global_acc = []
client_acc_matrix = []

# ---------------- federated training ----------------
for rnd in range(ROUNDS):
    print(f"Round {rnd + 1}")

    client_models = []
    round_client_acc = []

    for cid in client_ids:
        Xc = pd.read_csv(os.path.join(CLIENT_DIR, cid, "X.csv"))
        yc = pd.read_csv(os.path.join(CLIENT_DIR, cid, "y.csv")).values.ravel()

        Xc = Xc.select_dtypes(include=[np.number]).astype(np.float32)
        Xc = scaler.transform(Xc)

        local_model = copy.deepcopy(global_model)

        for _ in range(LOCAL_EPOCHS):
            local_model.partial_fit(Xc, yc)

        y_pred_local = local_model.predict(X_test)
        round_client_acc.append(accuracy_score(y_test, y_pred_local))

        client_models.append(local_model)

    # aggregate
    coef = np.mean([m.coef_ for m in client_models], axis=0)
    intercept = np.mean([m.intercept_ for m in client_models], axis=0)

    global_model.coef_ = coef
    global_model.intercept_ = intercept
    global_model.classes_ = client_models[0].classes_

    y_pred_global = global_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_global)

    global_acc.append(acc)
    client_acc_matrix.append(round_client_acc)

    print("Global accuracy:", round(acc, 4))
    print("Mean client accuracy:", round(np.mean(round_client_acc), 4))

# ---------------- save results ----------------
df = pd.DataFrame(
    client_acc_matrix,
    columns=[f"client_{i}" for i in range(num_clients)]
)

df.insert(0, "round", range(1, ROUNDS + 1))
df["mean_client_accuracy"] = df.iloc[:, 1:1 + num_clients].mean(axis=1)
df["min_client_accuracy"] = df.iloc[:, 1:1 + num_clients].min(axis=1)
df["max_client_accuracy"] = df.iloc[:, 1:1 + num_clients].max(axis=1)
df["global_accuracy"] = global_acc

df.to_csv(os.path.join(RESULTS_DIR, "federated_convergence.csv"), index=False)

print("Federated training completed")

