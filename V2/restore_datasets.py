dp_files = {}

dp_files["V2/datasets/labeling.py"] = '''import numpy as np

class GroundTruthLabeler:
    LABEL_NAMES = {0: "Healthy", 1: "Congested", 2: "Unhealthy"}
    def __init__(self, queue_congested_ratio=0.70, traffic_intensity_threshold=0.90, energy_unhealthy_ratio=0.15, plr_unhealthy_threshold=0.40):
        self.queue_thresh = queue_congested_ratio
        self.intensity_thresh = traffic_intensity_threshold
        self.energy_thresh = energy_unhealthy_ratio
        self.plr_thresh = plr_unhealthy_threshold

    def label_vector(self, row: np.ndarray) -> int:
        if row[0] <= self.energy_thresh or row[6] >= self.plr_thresh:
            return 2
        if row[2] >= self.queue_thresh or row[5] >= self.intensity_thresh:
            return 1
        return 0
'''

dp_files["V2/datasets/generator.py"] = '''import os, json, copy
import pandas as pd
from V2.simulator.network import WSNNetwork
from V2.datasets.labeling import GroundTruthLabeler

def default_routing(u_id, net):
    node = net.nodes[u_id]
    alive = [n for n in node.neighbors if net.nodes[n].is_alive]
    if not alive:
        return None
    return min(alive, key=lambda n: net.nodes[n].dist_to_sink)

class DatasetGenerator:
    def __init__(self, base_config_path="V2/configs/default_config.json", output_dir="V2/datasets"):
        self.base_config = json.load(open(base_config_path))
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
'''

dp_files["V2/datasets/__init__.py"] = '''from .labeling import GroundTruthLabeler
from .generator import DatasetGenerator
__all__ = ["GroundTruthLabeler", "DatasetGenerator"]
'''

dp_files["V2/preprocessing/temporal.py"] = '''import numpy as np
import pandas as pd
from typing import Tuple, List

class TemporalSequenceBuilder:
    def __init__(self, window_size: int = 5):
        self.window_size = int(window_size)

    def build_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "Node_Status"):
        X_seq_list, y_seq_list = [], []
        sort_cols = [c for c in ["Seed", "Node_ID", "Epoch"] if c in df.columns]
        df_sorted = df.sort_values(by=sort_cols).copy()
        for (_, _), group in df_sorted.groupby(["Seed", "Node_ID"]):
            feat = group[feature_cols].values.astype(np.float32)
            lbl = group[target_col].values.astype(np.int64)
            for t in range(len(feat)):
                if t + 1 < self.window_size:
                    pad = np.repeat(feat[0:1], self.window_size - (t + 1), axis=0)
                    win = np.vstack([pad, feat[:t+1]])
                else:
                    win = feat[t - self.window_size + 1 : t + 1]
                X_seq_list.append(win)
                y_seq_list.append(lbl[t])
        return np.array(X_seq_list, dtype=np.float32), np.array(y_seq_list, dtype=np.int64)
'''

dp_files["V2/preprocessing/graph_builder.py"] = '''import numpy as np
import torch

class GraphBuilder:
    @staticmethod
    def compute_normalized_adjacency(adj_matrix: np.ndarray) -> np.ndarray:
        N = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(N, dtype=np.float32)
        d_tilde = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(np.maximum(d_tilde, 1e-6), -0.5)
        D_inv = np.diag(d_inv_sqrt)
        return (D_inv @ A_tilde @ D_inv).astype(np.float32)
'''

for path, content in dp_files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Restored: {path}")
