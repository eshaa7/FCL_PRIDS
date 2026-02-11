import pandas as pd
import numpy as np

DATA_PATH = "data/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Initial shape:", df.shape)

# Drop duplicate rows
df = df.drop_duplicates()
print("After duplicate removal:", df.shape)

# Drop columns that leak labels or IDs
drop_cols = [
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Timestamp"
]

drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Encode target
df["Attack_label"] = df["Attack_label"].astype(int)

# Separate labels
y = df["Attack_label"]
X = df.drop(columns=["Attack_label", "Attack_type"])

# Handle categorical columns safely
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if X[c].nunique() < 20]

print("Encoding categorical columns:", cat_cols)

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Fill missing values
X = X.fillna(0)

print("Final feature shape:", X.shape)

# Save processed data
X.to_csv("data/X_processed.csv", index=False)
y.to_csv("data/y_labels.csv", index=False)

print("Preprocessing completed successfully")

