import os
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

ROUNDS = 5
BATCH_SIZE = 1000
RANDOM_STATE = 42

print("Loading dataset")

X = pd.read_csv(os.path.join(DATA_DIR, "X_processed.csv"))
y = pd.read_csv(os.path.join(DATA_DIR, "y_labels.csv")).values.ravel()

X = X.select_dtypes(include=[np.number]).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

model = SGDClassifier(
    loss="log_loss",
    learning_rate="optimal",
    max_iter=1,
    tol=None,
    class_weight=class_weights,
    random_state=RANDOM_STATE
)

print("Starting centralized iterative training")

global_acc = []

model.partial_fit(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], classes=classes)

for rnd in range(ROUNDS):
    print(f"Epoch {rnd + 1}")

    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_train[i:i + BATCH_SIZE]
        y_batch = y_train[i:i + BATCH_SIZE]
        model.partial_fit(X_batch, y_batch)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    global_acc.append(acc)

    print("Accuracy:", round(acc, 4))

pd.DataFrame({
    "round": range(1, ROUNDS + 1),
    "centralized_accuracy": global_acc
}).to_csv(
    os.path.join(RESULTS_DIR, "centralized_convergence.csv"),
    index=False
)

print("Centralized iterative training completed")

