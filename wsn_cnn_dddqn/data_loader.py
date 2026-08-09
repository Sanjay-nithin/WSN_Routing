"""
Module 1: Dataset Loading and Preprocessing
============================================
Loads all 5 xlsx files from 'new datasets/', cleans them, and returns
a unified per-node feature DataFrame ready for CNN training and DDDQN.

Dataset inventory (discovered by inspection):
  CrossLayer  : 5 000 rows, 99 unique nodes. Columns: Node_ID, Neighbor_Node,
                Node_Velocity, Residual_Energy, Initial_Energy, Link_Quality,
                Packet_Loss_Rate, Distance_to_NextHop, Delay, Throughput, Timestamp
  DS          : 374 661 rows, multi-node. Columns: id, Time, Rank,
                Expaned Energy, Dist_To_CH, dist_CH_To_BS, DATA_S, DATA_R,
                Data_Sent_To_BS, Attack type  [Label: Attack type]
  Latency     : 1 000 rows, 1 000 unique nodes. Columns: Node_ID, Hop_Count,
                Transmission_Delay, Buffer_Occupancy, Channel_Utilization,
                Energy_Level, Link_Quality, PDR, Traffic_Class, Latency_Category
                [Label: Latency_Category (Low/Medium/High Latency)]
  Project     : 1 000 rows. Columns: node_id, cluster_id, residual_energy,
                proximity_to_CH, total_energy_used, role, network_lifetime,
                data_packets_sent, data_packets_received
  Selected    : 10 000 rows, 10 000 unique nodes. Columns: Node_ID, Timestamp,
                Residual_Energy, Transmission_Power, Signal_Strength, Noise_Level,
                Energy_Consumption, Packet_Loss_Rate, Network_Lifetime,
                Temperature, Humidity
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "new datasets")

# ── column name aliases ─────────────────────────────────────────────────────
# Maps each file's actual column names to canonical WSN feature names used
# throughout the pipeline.

CROSSLAYER_COLS = {
    "Node_ID":              "node_id",
    "Neighbor_Node":        "neighbor_node",
    "Node_Velocity":        "node_velocity",
    "Residual_Energy":      "residual_energy",
    "Link_Quality":         "link_quality",
    "Packet_Loss_Rate":     "packet_loss_rate",
    "Distance_to_NextHop":  "distance_to_nexthop",
    "Delay":                "delay",
    "Throughput":           "throughput",
    "Timestamp":            "timestamp",
}

DS_COLS = {
    " id":                  "node_id",
    " Time":                "time",
    "Rank":                 "rank",
    "Expaned Energy":       "expended_energy",
    " Dist_To_CH":          "dist_to_ch",
    " dist_CH_To_BS":       "dist_ch_to_bs",
    " DATA_S":              "data_sent",
    " DATA_R":              "data_received",
    " Data_Sent_To_BS":     "data_sent_to_bs",
    "Attack type":          "attack_type",
}

LATENCY_COLS = {
    "Node_ID":              "node_id",
    "Hop_Count":            "hop_count",
    "Transmission_Delay":   "transmission_delay",
    "Buffer_Occupancy":     "buffer_occupancy",
    "Channel_Utilization":  "channel_utilization",
    "Energy_Level":         "energy_level",
    "Link_Quality":         "link_quality",
    "PDR":                  "pdr",
    "Traffic_Class":        "traffic_class",
    "Latency_Category":     "latency_category",
}

PROJECT_COLS = {
    "node_id":              "node_id",
    "cluster_id":           "cluster_id",
    "residual_energy":      "residual_energy",
    "proximity_to_CH":      "proximity_to_ch",
    "total_energy_used":    "total_energy_used",
    "role":                 "role",
    "network_lifetime":     "network_lifetime",
    "data_packets_sent":    "data_packets_sent",
    "data_packets_received":"data_packets_received",
}

SELECTED_COLS = {
    "Node_ID":              "node_id",
    "Residual_Energy":      "residual_energy",
    "Transmission_Power":   "transmission_power",
    "Signal_Strength":      "signal_strength",
    "Noise_Level":          "noise_level",
    "Energy_Consumption":   "energy_consumption",
    "Packet_Loss_Rate":     "packet_loss_rate",
    "Network_Lifetime":     "network_lifetime",
    "Temperature":          "temperature",
    "Humidity":             "humidity",
}


def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ── individual loaders ───────────────────────────────────────────────────────

def load_crosslayer() -> pd.DataFrame:
    df = pd.read_excel(_path("WSN_CrossLayer_Selected_Features.xlsx"))
    df = df.rename(columns={k: v for k, v in CROSSLAYER_COLS.items() if k in df.columns})
    df = df.drop(columns=["Initial_Energy"], errors="ignore")
    return df


def load_ds(sample_n: int = 50_000) -> pd.DataFrame:
    """DS dataset is large (374 k rows). We sample for tractability."""
    df = pd.read_excel(_path("WSN_DS_Selected_Features.xlsx"))
    df.columns = df.columns.str.strip()
    col_map = {k.strip(): v for k, v in DS_COLS.items()}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # Normalise node IDs (format 101000 → sequential int)
    df["node_id"] = df["node_id"].astype(str)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


def load_latency() -> pd.DataFrame:
    df = pd.read_excel(_path("WSN_Latency_Selected_Features.xlsx"))
    df = df.rename(columns={k: v for k, v in LATENCY_COLS.items() if k in df.columns})
    return df


def load_project() -> pd.DataFrame:
    df = pd.read_excel(_path("WSN_Project_Important_Features(1).xlsx"))
    df = df.rename(columns={k: v for k, v in PROJECT_COLS.items() if k in df.columns})
    return df


def load_selected() -> pd.DataFrame:
    df = pd.read_excel(_path("WSN_Selected_Features(3).xlsx"))
    df = df.rename(columns={k: v for k, v in SELECTED_COLS.items() if k in df.columns})
    df = df.drop(columns=["Timestamp"], errors="ignore")
    return df


# ── Rule-based labeling ──────────────────────────────────────────────────────
# Datasets that already carry explicit labels:
#   • DS (Attack type): Normal / Flooding / TDMA / Grayhole / Blackhole
#   • Latency (Latency_Category): Low Latency / Medium Latency / High Latency
#
# Datasets without labels (CrossLayer, Project, Selected) are assigned
# three-class node-condition labels using the rule-based strategy below.
#
#  HEALTHY    : residual_energy > 0.6, packet_loss_rate < 0.08,
#               link_quality > 0.7 (or pdr > 0.85, if available)
#  CONGESTED  : buffer_occupancy > 70 OR packet_loss_rate > 0.15
#               OR channel_utilization > 75
#  UNHEALTHY  : everything else (low energy, medium loss, poor link)
#
# The label encoding is:  0=Healthy, 1=Congested, 2=Unhealthy

LABEL_NAMES = {0: "Healthy", 1: "Congested", 2: "Unhealthy"}


def _rule_label_crosslayer(df: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(df), 2, dtype=int)  # default = Unhealthy
    healthy = (
        (df["residual_energy"] > 0.6) &
        (df["packet_loss_rate"] < 0.08) &
        (df["link_quality"] > 0.7)
    )
    congested = (
        (df["packet_loss_rate"] > 0.15) |
        (df["link_quality"] < 0.55)
    )
    labels[congested.values] = 1
    labels[healthy.values] = 0
    return labels


def _rule_label_selected(df: pd.DataFrame) -> np.ndarray:
    # Residual_Energy range is 0–10 J in this dataset
    labels = np.full(len(df), 2, dtype=int)
    e_max = df["residual_energy"].max()
    e_norm = df["residual_energy"] / e_max
    plr_norm = df["packet_loss_rate"] / df["packet_loss_rate"].max()

    healthy = (e_norm > 0.5) & (plr_norm < 0.15)
    congested = (plr_norm > 0.6) | (e_norm < 0.1)
    labels[congested.values] = 1
    labels[healthy.values] = 0
    return labels


def _rule_label_project(df: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(df), 2, dtype=int)
    # Use delivery ratio proxy = received / sent (clipped)
    sent = df["data_packets_sent"].replace(0, 1)
    recv = df["data_packets_received"]
    pdr = (recv / sent).clip(0, 1)
    e_norm = df["residual_energy"] / df["residual_energy"].max()

    healthy = (pdr > 0.8) & (e_norm > 0.6)
    congested = (pdr < 0.4) | (e_norm < 0.25)
    labels[congested.values] = 1
    labels[healthy.values] = 0
    return labels


def _map_latency_to_3class(df: pd.DataFrame) -> np.ndarray:
    """Map Latency_Category → 0/1/2."""
    mapping = {
        "Low Latency":    0,   # Healthy
        "Medium Latency": 2,   # Unhealthy
        "High Latency":   1,   # Congested
    }
    return df["latency_category"].map(mapping).fillna(2).astype(int).values


def _map_attack_to_3class(df: pd.DataFrame) -> np.ndarray:
    """Map Attack type → 0/1/2 for CNN training on DS data."""
    mapping = {
        "Normal":    0,   # Healthy
        "Flooding":  1,   # Congested (bandwidth attack)
        "TDMA":      1,   # Congested (timing disruption)
        "Grayhole":  2,   # Unhealthy (selective drop)
        "Blackhole": 2,   # Unhealthy (full drop)
    }
    return df["attack_type"].map(mapping).fillna(2).astype(int).values


# ── CNN feature extraction ───────────────────────────────────────────────────

CNN_FEATURES_CROSSLAYER = [
    "node_velocity", "residual_energy", "link_quality",
    "packet_loss_rate", "distance_to_nexthop", "delay", "throughput",
]

CNN_FEATURES_DS = [
    "expended_energy", "dist_to_ch", "dist_ch_to_bs",
    "data_sent", "data_received", "data_sent_to_bs",
]

CNN_FEATURES_LATENCY = [
    "hop_count", "transmission_delay", "buffer_occupancy",
    "channel_utilization", "energy_level", "link_quality", "pdr",
]

CNN_FEATURES_PROJECT = [
    "residual_energy", "proximity_to_ch", "total_energy_used",
    "network_lifetime", "data_packets_sent", "data_packets_received",
]

CNN_FEATURES_SELECTED = [
    "residual_energy", "transmission_power", "signal_strength",
    "noise_level", "energy_consumption", "packet_loss_rate",
    "network_lifetime", "temperature", "humidity",
]


def build_cnn_dataset() -> dict:
    """
    Returns a dict with keys:
      X       : np.ndarray (N, max_features) – padded feature matrix
      y       : np.ndarray (N,)              – 0/1/2 labels
      scaler  : fitted MinMaxScaler
      feature_names : list[str]
    """
    print("[DataLoader] Loading all datasets …")
    dfs_X, dfs_y, feat_cols = [], [], []

    # --- CrossLayer ---
    cl = load_crosslayer()
    cl_y = _rule_label_crosslayer(cl)
    cl_X = cl[CNN_FEATURES_CROSSLAYER].fillna(0).values
    dfs_X.append(cl_X); dfs_y.append(cl_y)
    if not feat_cols:
        feat_cols = CNN_FEATURES_CROSSLAYER
    print(f"  CrossLayer : {cl_X.shape[0]:>6} rows, {cl_X.shape[1]} features")

    # --- DS ---
    ds = load_ds(sample_n=15_000)
    ds_y = _map_attack_to_3class(ds)
    ds_X = ds[CNN_FEATURES_DS].fillna(0).values
    # Pad to same width as CrossLayer (7 cols) with zeros
    _pad = max(0, cl_X.shape[1] - ds_X.shape[1])
    ds_X = np.pad(ds_X, ((0, 0), (0, _pad)))
    dfs_X.append(ds_X); dfs_y.append(ds_y)
    print(f"  DS         : {ds_X.shape[0]:>6} rows, {ds_X.shape[1]} features (padded)")

    # --- Latency ---
    lt = load_latency()
    lt_y = _map_latency_to_3class(lt)
    lt_X = lt[CNN_FEATURES_LATENCY].fillna(0).values
    _pad = max(0, cl_X.shape[1] - lt_X.shape[1])
    lt_X = np.pad(lt_X, ((0, 0), (0, _pad)))
    dfs_X.append(lt_X); dfs_y.append(lt_y)
    print(f"  Latency    : {lt_X.shape[0]:>6} rows, {lt_X.shape[1]} features (padded)")

    # --- Project ---
    pr = load_project()
    pr_y = _rule_label_project(pr)
    pr_X = pr[CNN_FEATURES_PROJECT].fillna(0).values
    _pad = max(0, cl_X.shape[1] - pr_X.shape[1])
    pr_X = np.pad(pr_X, ((0, 0), (0, _pad)))
    dfs_X.append(pr_X); dfs_y.append(pr_y)
    print(f"  Project    : {pr_X.shape[0]:>6} rows, {pr_X.shape[1]} features (padded)")

    # --- Selected ---
    sl = load_selected()
    sl_y = _rule_label_selected(sl)
    sl_X = sl[CNN_FEATURES_SELECTED].fillna(0).values
    # Selected has 9 cols — take first 7 for alignment
    sl_X = sl_X[:, :cl_X.shape[1]]
    dfs_X.append(sl_X); dfs_y.append(sl_y)
    print(f"  Selected   : {sl_X.shape[0]:>6} rows, {sl_X.shape[1]} features")

    X = np.vstack(dfs_X).astype(np.float32)
    y = np.concatenate(dfs_y).astype(np.int64)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    counts = {LABEL_NAMES[k]: int((y == k).sum()) for k in [0, 1, 2]}
    print(f"  Total rows : {len(X)}")
    print(f"  Label dist : {counts}")

    return {
        "X": X,
        "y": y,
        "scaler": scaler,
        "feature_names": feat_cols,
        "n_features": X.shape[1],
    }


def build_routing_env_data() -> dict:
    """
    Returns the CrossLayer dataframe enriched with neighbor topology
    for use by the DDDQN WSN environment.
    """
    df = load_crosslayer()
    return df
