import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from V2.simulator.network import WSNNetwork
from V2.routing.cost_functions import RoutingCostFunction
class LocalizedMultiMetricRouter:
    def __init__(self, cost_function: RoutingCostFunction, ml_model: Optional[Any] = None, preprocessor: Optional[Any] = None):
        self.cost_fn = cost_function
        self.ml_model = ml_model
        self.preprocessor = preprocessor

    def get_next_hop(self, u_id: int, network: WSNNetwork) -> Optional[int]:
        u_node = network.nodes[u_id]
        alive = [v for v in u_node.neighbors if network.nodes[v].is_alive or v == network.sink_id]
        if not alive:
            return None
        return min(alive, key=lambda v: self.cost_fn.compute_edge_cost(
            distance=u_node.neighbors[v]["distance"],
            tx_range=network.tx_range,
            receiver_queue_ratio=network.nodes[v].queue_occupancy if not network.nodes[v].is_sink else 0.0,
            receiver_energy_ratio=network.nodes[v].energy.energy_ratio if not network.nodes[v].is_sink else 1.0,
            link_etx=u_node.neighbors[v].get("etx", 1.0)
        ))

class WeightOptimizer:
    def __init__(self, base_config: Dict[str, Any], ml_model: Optional[Any] = None, preprocessor: Optional[Any] = None):
        self.base_config = copy.deepcopy(base_config)
        self.ml_model = ml_model
        self.preprocessor = preprocessor

    def evaluate_weight_vector(self, weights: Tuple[float, float, float, float], sim_duration_s: float = 15.0, dt: float = 0.2) -> Dict[str, Any]:
        alpha, beta, gamma, eta = weights
        cost_fn = RoutingCostFunction(alpha_dist=alpha, beta_traffic=beta, gamma_energy=gamma, delta_etx=0.0, eta_ml=eta)
        router = LocalizedMultiMetricRouter(cost_function=cost_fn, ml_model=self.ml_model, preprocessor=self.preprocessor)
        cfg = copy.deepcopy(self.base_config)
        cfg["network"]["num_nodes"] = 30
        cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = 3.0
        net = WSNNetwork(cfg)
        steps = int(sim_duration_s / dt)
        for s in range(steps):
            net.step(dt=dt, routing_function=router.get_next_hop)
        metrics = net.compute_metrics()
        return {
            "alpha": alpha, "beta": beta, "gamma": gamma, "eta": eta,
            "pdr_percent": metrics["pdr_percent"], "avg_delay_s": metrics["avg_delay_s"],
            "total_energy_j": metrics["total_energy_j"], "dead_node_count": metrics["dead_node_count"]
        }

    def run_dirichlet_search(self, num_samples: int = 25, seed: int = 42) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        samples = rng.dirichlet(np.ones(4), size=num_samples)
        results = []
        for row in samples:
            weights = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            res = self.evaluate_weight_vector(weights)
            results.append(res)
        return results

    @staticmethod
    def filter_pareto_optimal(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pareto = []
        for i, cand in enumerate(results):
            dominated = False
            for j, other in enumerate(results):
                if i == j:
                    continue
                if (
                    other["pdr_percent"] >= cand["pdr_percent"] and
                    other["avg_delay_s"] <= cand["avg_delay_s"] and
                    other["total_energy_j"] <= cand["total_energy_j"] and
                    other["dead_node_count"] <= cand["dead_node_count"]
                ):
                    if (
                        other["pdr_percent"] > cand["pdr_percent"] or
                        other["avg_delay_s"] < cand["avg_delay_s"] or
                        other["total_energy_j"] < cand["total_energy_j"] or
                        other["dead_node_count"] < cand["dead_node_count"]
                    ):
                        dominated = True
                        break
            if not dominated:
                pareto.append(cand)
        return pareto
