"""
Normalized Parametric Link Cost Functions for WSN Routing.
"""
from typing import Dict, Any, Optional
import numpy as np

class RoutingCostFunction:
    def __init__(
        self,
        alpha_dist: float = 0.25,
        beta_traffic: float = 0.25,
        gamma_energy: float = 0.25,
        delta_etx: float = 0.0,
        eta_ml: float = 0.25,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        self.alpha = float(alpha_dist)
        self.beta = float(beta_traffic)
        self.gamma = float(gamma_energy)
        self.delta = float(delta_etx)
        self.eta = float(eta_ml)

        total_w = self.alpha + self.beta + self.gamma + self.delta + self.eta
        if total_w > 0:
            self.alpha /= total_w
            self.beta /= total_w
            self.gamma /= total_w
            self.delta /= total_w
            self.eta /= total_w

        self.risk_w = risk_weights or {"w_c": 0.8, "w_u": 2.0}

    def compute_edge_cost(
        self,
        distance: float,
        tx_range: float,
        receiver_queue_ratio: float,
        receiver_energy_ratio: float,
        link_etx: float = 1.0,
        ml_probs: Optional[np.ndarray] = None
    ) -> float:
        norm_dist = min(1.0, max(0.0, distance / max(1.0, tx_range)))
        norm_queue = min(1.0, max(0.0, receiver_queue_ratio))
        norm_energy_dep = min(1.0, max(0.0, 1.0 - receiver_energy_ratio))
        norm_etx = min(1.0, max(0.0, (link_etx - 1.0) / 19.0))

        if ml_probs is not None and len(ml_probs) == 3:
            p_c = float(ml_probs[1])
            p_u = float(ml_probs[2])
            w_c = self.risk_w.get("w_c", 0.8)
            w_u = self.risk_w.get("w_u", 2.0)
            norm_risk = min(1.0, max(0.0, (w_c * p_c + w_u * p_u) / max(1e-6, w_u)))
        else:
            norm_risk = 0.0

        cost = (
            self.alpha * norm_dist +
            self.beta * norm_queue +
            self.gamma * norm_energy_dep +
            self.delta * norm_etx +
            self.eta * norm_risk
        )
        return max(1e-4, float(cost))
