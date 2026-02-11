import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("Loading processed data")

X = pd.read_csv("data/X_processed.csv", low_memory=False)
y = pd.read_csv("data/y_labels.csv")
X = X.select_dtypes(include=["number"])
y = y.values.ravel()

print("Data shape:", X.shape, y.shape)

print("Train test split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Scaling features")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training centralized model")
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Evaluating model")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Centralized Accuracy:", acc)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))

os.makedirs("results", exist_ok=True)
joblib.dump(model, "results/centralized_model.joblib")
joblib.dump(scaler, "results/scaler.joblib")

print("Centralized training completed")

import pandas as pd
pd.DataFrame(
    {"model": ["centralized"], "accuracy": [acc]}
).to_csv("results/centralized_accuracy.csv", index=False)

