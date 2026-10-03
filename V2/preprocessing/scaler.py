import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        self.scaler.fit(X)
        self.is_fitted = True
        if feature_names:
            self.feature_names = list(feature_names)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)

    def save(self, filepath: str = "V2/preprocessing/scaler.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, filepath)

    def load(self, filepath: str = "V2/preprocessing/scaler.joblib"):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", [])
        self.is_fitted = True
        return self
