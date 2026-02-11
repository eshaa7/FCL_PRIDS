# FCL-PRIDS: Federated Continual Learning for Privacy-Preserving Intrusion Detection

![Framework](https://img.shields.io/badge/Framework-Federated_Learning-blue)
![Model](https://img.shields.io/badge/Model-1D_CNN-orange)
![Dataset](https://img.shields.io/badge/Dataset-Edge_IIoTset-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-lightgrey)

## 📖 Abstract
This repository implements **FCL-PRIDS**, a framework designed to secure Industrial IoT (IIoT) networks against evolving cyber threats. Unlike traditional centralized IDSs, this project employs **Federated Continual Learning (FCL)** to train a global model across distributed edge devices without sharing raw traffic data.

The core classification engine is a **1D-Convolutional Neural Network (1D-CNN)**, optimized for extracting spatial features from time-series network traffic (MQTT, CoAP, HTTP).

## 📂 Repository Structure
```text
.
├── src/
│   ├── FCL_PRIDS_CNN.py            # The 1D-CNN model architecture
│   ├── train_federated.py          # FL training loop (Server/Client logic)
│   ├── train_centralized.py        # Baseline centralized training for comparison
│   ├── compute_communication_cost.py # Scripts to measure bandwidth usage
│   └── split_clients.py            # Data partitioning logic for non-IID settings
├── results/
│   ├── accuracy_comparison.png     # FCL vs Centralized performance
│   └── centralized_model.joblib    # Saved model weights
├── data/                           # (Excluded via .gitignore)
└── requirements.txt                # Python dependencies
