# WSN Node-Condition Classifier: 1-D CNN

An end-to-end machine learning pipeline for Wireless Sensor Network (WSN) node-health classification. A 1-D Convolutional Neural Network (CNN) classifies each sensor node's condition from real WSN telemetry into three classes: **Healthy**, **Congested**, or **Unhealthy**.

---

## Architecture

```
new datasets/*.xlsx
        │
        ▼
  DataLoader & Preprocessor
  (data_loader.py)
        │  unified feature matrix (32 000 samples × 7 features)
        │  + rule-based / existing labels (Healthy / Congested / Unhealthy)
        ▼
  1-D CNN Classifier
  (cnn_model.py + cnn_train.py)
        │  per-node condition label + probabilities [P_H, P_C, P_U]
        ▼
  Evaluation & Visualization
  (visualization.py)
        │  4 plots + JSON metrics
        ▼
  outputs/
```

---

## Datasets

All five `.xlsx` files in `new datasets/` are used automatically. No manual preprocessing is required.

| File | Rows | Nodes | Key Features | Labels |
|------|-----:|------:|-------------|--------|
| `WSN_CrossLayer_Selected_Features` | 5 000 | 99 | Residual Energy, Link Quality, Packet Loss, Delay, Throughput, Velocity, Distance | Rule-based |
| `WSN_DS_Selected_Features` | 374 661 | multi | Expended Energy, Dist to CH/BS, Data Sent/Received | ✅ `Attack type` (Normal / Flooding / TDMA / Grayhole / Blackhole) |
| `WSN_Latency_Selected_Features` | 1 000 | 1 000 | Hop Count, Transmission Delay, Buffer Occupancy, Channel Utilization, Energy Level, PDR | ✅ `Latency_Category` (Low / Medium / High) |
| `WSN_Project_Important_Features(1)` | 1 000 | 1 000 | Residual Energy, Proximity to CH, Network Lifetime, Packets Sent/Received | Rule-based |
| `WSN_Selected_Features(3)` | 10 000 | 10 000 | Residual Energy, Signal Strength, Energy Consumption, Packet Loss, Network Lifetime | Rule-based |

**DS** is sampled to 15 000 rows for tractability. Total merged dataset: **32 000 samples**.

### Rule-based Labeling Strategy

For datasets without existing labels, the following rules assign 3-class node-condition labels:

| Class | Rule |
|-------|------|
| **Healthy** (0) | `residual_energy > 0.6` AND `packet_loss_rate < 0.08` AND `link_quality > 0.7` |
| **Congested** (1) | `packet_loss_rate > 0.15` OR `link_quality < 0.55` |
| **Unhealthy** (2) | All remaining nodes (moderate loss, low energy, degraded links) |

Existing labels are mapped as: DS `Normal`→Healthy, `Flooding/TDMA`→Congested, `Grayhole/Blackhole`→Unhealthy; Latency `Low Latency`→Healthy, `High Latency`→Congested, `Medium Latency`→Unhealthy.

---

## CNN Classifier

**Architecture**: 1-D CNN treating the feature vector as a 1-D signal.

```
Input (B, 7)
  → unsqueeze → (B, 1, 7)
  → Conv1d(1→32, k=3) + BN + ReLU
  → Conv1d(32→64, k=3) + BN + ReLU
  → Conv1d(64→128, k=3) + BN + ReLU
  → GlobalAvgPool → (B, 128)
  → Dropout(0.3)
  → FC(128→64) + ReLU
  → FC(64→3)
Output: logits for [Healthy, Congested, Unhealthy]
```

**Training details**:
- Loss: Cross-Entropy with class weights (handles imbalance)
- Optimizer: Adam with weight decay 1e-4
- Scheduler: Cosine Annealing
- Split: 80% train / 20% test
- **Achieved test accuracy: ~90%**

---

## Project Structure

```
WSN/
├── main.ipynb                          ← End-to-end notebook (run this)
├── README.md                           ← This file
│
├── wsn_cnn_dddqn/
│   ├── __init__.py
│   ├── data_loader.py                  ← Load all 5 xlsx + labeling logic
│   ├── cnn_model.py                    ← 1-D CNN architecture
│   ├── cnn_train.py                    ← CNN training + evaluation
│   └── visualization.py               ← All plots + metrics JSON
│
├── new datasets/
│   ├── WSN_CrossLayer_Selected_Features.xlsx
│   ├── WSN_DS_Selected_Features.xlsx
│   ├── WSN_Latency_Selected_Features.xlsx
│   ├── WSN_Project_Important_Features(1).xlsx
│   └── WSN_Selected_Features(3).xlsx
│
└── outputs/                            ← Created automatically
    ├── cnn_model.pt
    ├── feature_scaler.pkl
    ├── cnn_training_curves.png
    ├── cnn_confusion_matrix.png
    ├── cnn_per_class_metrics.png
    ├── node_condition_heatmap.png
    └── cnn_evaluation_results.json
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib seaborn tqdm openpyxl
```

### 2. Run the notebook
Open `main.ipynb` in Jupyter and run all cells top-to-bottom.

### 3. Run as a script (alternative)
```python
import sys; sys.path.insert(0, '.')
from wsn_cnn_dddqn.data_loader import build_cnn_dataset
from wsn_cnn_dddqn.cnn_train import train_cnn
from wsn_cnn_dddqn.visualization import (
    plot_cnn_training, plot_confusion_matrix,
    plot_cnn_metrics, plot_node_condition_heatmap,
    save_metrics_table,
)

# 1. Build dataset
data = build_cnn_dataset()

# 2. Train CNN
cnn_results = train_cnn(data['X'], data['y'], data['scaler'])

# 3. Generate plots & save metrics
plot_cnn_training(cnn_results['history'])
plot_confusion_matrix(cnn_results['cm'])
plot_cnn_metrics(cnn_results['report'])
save_metrics_table(cnn_results)
```

---

## Outputs

| File | Description |
|------|-------------|
| `cnn_model.pt` | Trained CNN state dict |
| `feature_scaler.pkl` | Fitted MinMaxScaler for inference |
| `cnn_training_curves.png` | Loss and accuracy per epoch |
| `cnn_confusion_matrix.png` | Normalized confusion matrix |
| `cnn_per_class_metrics.png` | Precision / Recall / F1 per class |
| `node_condition_heatmap.png` | CNN class probability heatmap per node |
| `cnn_evaluation_results.json` | All numeric metrics in machine-readable form |

---

## Requirements

| Package | Version tested |
|---------|---------------|
| Python | 3.12.7 |
| PyTorch | 2.13.0+cpu |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| tqdm | 4.70.0 |
| openpyxl | 3.1.5 |
