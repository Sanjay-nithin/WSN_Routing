import numpy as np
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
