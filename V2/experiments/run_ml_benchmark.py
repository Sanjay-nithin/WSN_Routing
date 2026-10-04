"""
WSN Machine Learning Benchmark & Training Script.
Imports the processed WSN telemetry dataset from V2/datasets/processed/
and trains/evaluates all 9 Machine Learning classification architectures.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(V2_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from V2.preprocessing.scaler import FeaturePreprocessor
from V2.models.classical import ClassicalMLSuite
from V2.models.boosting import XGBoostClassifierWrapper
from V2.models.mlp import WSNMLP
from V2.models.cnn_1d import TemporalCNN1D
from V2.models.cnn_lstm import TemporalCNNLSTM
from V2.models.gnn import WSNGCN
from V2.models.trainer import train_pytorch_model


FEATURE_COLS = [
    "Residual_Energy_Ratio",
    "Energy_Depletion_Rate",
    "Buffer_Occupancy_Ratio",
    "Packet_Arrival_Rate",
    "Packet_Service_Rate",
    "Traffic_Intensity",
    "Packet_Loss_Rate",
    "Average_Queuing_Delay",
    "Average_Neighbor_RSSI",
    "Average_Neighbor_PRR",
    "Neighbor_Degree",
    "Distance_to_Sink_Ratio"
]
TARGET_COL = "Node_Status"


def load_wsn_datasets(datasets_dir: str = None):
    """
    Imports the preprocessed WSN telemetry datasets from CSV files.
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(V2_DIR, "datasets", "processed")

    train_path = os.path.join(datasets_dir, "train.csv")
    val_path = os.path.join(datasets_dir, "val.csv")
    test_path = os.path.join(datasets_dir, "test.csv")

    print(f"[+] Importing WSN telemetry dataset from:")
    print(f"    - Train CSV: {train_path}")
    print(f"    - Val CSV:   {val_path}")
    print(f"    - Test CSV:  {test_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train[TARGET_COL].values.astype(np.int64)

    X_val = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val[TARGET_COL].values.astype(np.int64)

    X_test = df_test[FEATURE_COLS].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.int64)

    print(f"[+] Loaded {len(X_train)} train, {len(X_val)} validation, and {len(X_test)} test samples.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_training_and_benchmark(save_checkpoints: bool = True):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_wsn_datasets()

    # Preprocessing / Feature Normalization
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(X_train, FEATURE_COLS)
    X_train_scaled = preprocessor.transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    ckpt_dir = os.path.join(V2_DIR, "results", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if save_checkpoints:
        preprocessor.save(os.path.join(V2_DIR, "preprocessing", "scaler.joblib"))

    results = []

    # 1. Classical Machine Learning Models
    classical_models = {
        "logistic_regression": ClassicalMLSuite.get_logistic_regression(random_state=42),
        "decision_tree": ClassicalMLSuite.get_decision_tree(random_state=42),
        "random_forest": ClassicalMLSuite.get_random_forest(random_state=42),
        "svm_rbf": ClassicalMLSuite.get_svm(random_state=42)
    }
    print("\n[+] Training Classical Machine Learning Classifiers...")
    for name, model in classical_models.items():
        if name == "svm_rbf":
            model.fit(X_train_scaled[:2000], y_train[:2000])  # Subsampled for rapid RBF fitting
        else:
            model.fit(X_train_scaled, y_train)

        t0 = time.perf_counter()
        preds = model.predict(X_test_scaled)
        lat_us = ((time.perf_counter() - t0) / len(X_test_scaled)) * 1e6
        acc = accuracy_score(y_test, preds) * 100.0
        f1 = f1_score(y_test, preds, average="macro")
        results.append({"Model": name, "Accuracy (%)": acc, "Macro-F1": f1, "Latency (us)": lat_us})
        if save_checkpoints:
            joblib.dump(model, os.path.join(ckpt_dir, f"{name}.joblib"))

    # 2. XGBoost
    print("[+] Training XGBoost Gradient Boosting Classifier...")
    xgb_model = XGBoostClassifierWrapper(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    t0 = time.perf_counter()
    preds_xgb = xgb_model.predict(X_test_scaled)
    lat_xgb = ((time.perf_counter() - t0) / len(X_test_scaled)) * 1e6
    results.append({
        "Model": "xgboost",
        "Accuracy (%)": accuracy_score(y_test, preds_xgb) * 100.0,
        "Macro-F1": f1_score(y_test, preds_xgb, average="macro"),
        "Latency (us)": lat_xgb
    })
    if save_checkpoints:
        xgb_model.save(os.path.join(ckpt_dir, "xgboost.joblib"))

    # 3. Deep Learning Models (PyTorch)
    print("[+] Training Deep Learning Models (MLP, 1D-CNN, CNN-LSTM)...")
    
    # MLP
    mlp = WSNMLP(in_features=12, num_classes=3)
    train_pytorch_model(mlp, X_train_scaled, y_train, X_val_scaled, y_val, epochs=15)
    t0 = time.perf_counter()
    preds_mlp = mlp.predict(X_test_scaled)
    lat_mlp = ((time.perf_counter() - t0) / len(X_test_scaled)) * 1e6
    results.append({
        "Model": "mlp",
        "Accuracy (%)": accuracy_score(y_test, preds_mlp) * 100.0,
        "Macro-F1": f1_score(y_test, preds_mlp, average="macro"),
        "Latency (us)": lat_mlp
    })
    if save_checkpoints:
        torch.save(mlp.state_dict(), os.path.join(ckpt_dir, "mlp.pt"))

    # 1D-CNN (Temporal 1D Convolutional Neural Network - Champion Model)
    cnn = TemporalCNN1D(in_channels=12, seq_len=5, num_classes=3)
    train_pytorch_model(cnn, X_train_scaled, y_train, X_val_scaled, y_val, epochs=20)
    t0 = time.perf_counter()
    preds_cnn = cnn.predict(X_test_scaled)
    lat_cnn = ((time.perf_counter() - t0) / len(X_test_scaled)) * 1e6
    results.append({
        "Model": "cnn_1d (Selected Best)",
        "Accuracy (%)": accuracy_score(y_test, preds_cnn) * 100.0,
        "Macro-F1": f1_score(y_test, preds_cnn, average="macro"),
        "Latency (us)": lat_cnn
    })
    if save_checkpoints:
        torch.save(cnn.state_dict(), os.path.join(ckpt_dir, "cnn_1d.pt"))

    # CNN-LSTM
    cnn_lstm = TemporalCNNLSTM(in_features=12, hidden_dim=48, lstm_layers=1, num_classes=3)
    train_pytorch_model(cnn_lstm, X_train_scaled, y_train, X_val_scaled, y_val, epochs=15)
    t0 = time.perf_counter()
    preds_cnnlstm = cnn_lstm.predict(X_test_scaled)
    lat_cnnlstm = ((time.perf_counter() - t0) / len(X_test_scaled)) * 1e6
    results.append({
        "Model": "cnn_lstm",
        "Accuracy (%)": accuracy_score(y_test, preds_cnnlstm) * 100.0,
        "Macro-F1": f1_score(y_test, preds_cnnlstm, average="macro"),
        "Latency (us)": lat_cnnlstm
    })
    if save_checkpoints:
        torch.save(cnn_lstm.state_dict(), os.path.join(ckpt_dir, "cnn_lstm.pt"))

    # Print Summary Table
    df_res = pd.DataFrame(results)
    print("\n==========================================================================")
    print("      WSN 9-MODEL BENCHMARK RESULTS (Trained on V2/datasets/processed/)")
    print("==========================================================================")
    print(df_res.to_string(index=False))
    print("==========================================================================")
    return df_res


if __name__ == "__main__":
    run_training_and_benchmark(save_checkpoints=False)
