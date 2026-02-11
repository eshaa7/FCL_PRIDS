import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import deque

# --- STEP 1: LOAD AND CLEAN ---
def preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Remove metadata that causes bias
    drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'icmp.transmit_timestamp']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Remove rows with empty values
    df.dropna(inplace=True)
    
    # Scale features to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features = df.drop(columns=['Attack_type', 'Label'])
    scaled_features = scaler.fit_transform(features)
    
    # Encode labels to numbers
    le = LabelEncoder()
    labels = le.fit_transform(df['Attack_type'])
    
    return torch.tensor(scaled_features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# --- STEP 2: THE 1D-CNN ARCHITECTURE ---
class FCL_PRIDS_CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCL_PRIDS_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_dim // 2), 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- STEP 3: THE REPLAY BUFFER (Objective IV) ---
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, data, label):
        self.buffer.append((data, label))
# --- STEP 4: EXECUTION ---
if __name__ == "__main__":
    # 1. Provide the path to your dataset file here
    dataset_path = "data/Edge-IIoTset_datasetFL.csv" 
    
    print(f"--- Loading and Preprocessing: {dataset_path} ---")
    try:
        X, y = preprocess_data(dataset_path)
        print(f"Success! Data Shape: {X.shape}")
        
        # 2. Initialize the Model
        input_dim = X.shape[1]
        num_classes = len(torch.unique(y))
        model = FCL_PRIDS_CNN(input_dim, num_classes)
        
        print(f"Model initialized for {num_classes} attack types.")
        print("Ready for Federated Training...")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{dataset_path}'. Make sure the file is in the FCL_PRIDS folder.")
    except Exception as e:
        print(f"An error occurred: {e}")
