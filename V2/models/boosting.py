import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class XGBoostClassifierWrapper:
    def __init__(self, n_estimators: int = 80, max_depth: int = 5, learning_rate: float = 0.08, subsample: float = 0.85, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, objective="multi:softprob", num_class=3, random_state=random_state, n_jobs=-1, eval_metric="mlogloss"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        sample_weights = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
