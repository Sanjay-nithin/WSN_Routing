import os

routing_files = {}

routing_files["V2/routing/cost_functions.py"] = '''"""
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
'''

routing_files["V2/routing/hysteresis.py"] = '''from typing import Dict, Optional

class RoutingHysteresisFilter:
    def __init__(self, hysteresis_factor: float = 0.15):
        self.hysteresis = float(hysteresis_factor)
        self.active_routes: Dict[int, int] = {}
        self.active_costs: Dict[int, float] = {}
        self.route_changes_count = 0

    def should_update_route(
        self,
        node_id: int,
        candidate_next_hop: int,
        candidate_cost: float,
        is_current_alive: bool = True
    ) -> bool:
        if node_id not in self.active_routes:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            self.route_changes_count += 1
            return True

        current_hop = self.active_routes[node_id]
        current_cost = self.active_costs[node_id]

        if not is_current_alive or candidate_next_hop == current_hop:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            return True

        threshold_cost = current_cost * (1.0 - self.hysteresis)
        if candidate_cost < threshold_cost:
            self.active_routes[node_id] = candidate_next_hop
            self.active_costs[node_id] = candidate_cost
            self.route_changes_count += 1
            return True

        return False

    def get_active_hop(self, node_id: int) -> Optional[int]:
        return self.active_routes.get(node_id)
'''

routing_files["V2/routing/dijkstra.py"] = '''import heapq
from typing import Dict, Optional, Any
import numpy as np
from V2.simulator.network import WSNNetwork
from V2.routing.cost_functions import RoutingCostFunction
from V2.routing.hysteresis import RoutingHysteresisFilter

class DynamicDijkstraRouter:
    def __init__(
        self,
        cost_function: RoutingCostFunction,
        ml_model: Optional[Any] = None,
        hysteresis_filter: Optional[RoutingHysteresisFilter] = None,
        preprocessor: Optional[Any] = None
    ):
        self.cost_fn = cost_function
        self.ml_model = ml_model
        self.hysteresis = hysteresis_filter
        self.preprocessor = preprocessor
        self.routing_table: Dict[int, int] = {}
        self.path_costs: Dict[int, float] = {}

    def update_routing_table(self, network: WSNNetwork) -> None:
        sink_id = network.sink_id
        num_nodes = network.num_nodes

        node_probs = {}
        if self.ml_model is not None:
            features, _ = network.collect_dataset_sample()
            if self.preprocessor is not None:
                features = self.preprocessor.transform(features)
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)
                for i in range(num_nodes):
                    node_probs[i] = probs[i]

        dist = {i: float("inf") for i in range(num_nodes + 1)}
        next_hop = {i: None for i in range(num_nodes + 1)}
        dist[sink_id] = 0.0

        rev_adj = {i: [] for i in range(num_nodes + 1)}
        for u in range(num_nodes):
            u_node = network.nodes[u]
            if not u_node.is_alive:
                continue

            for v in u_node.neighbors:
                v_node = network.nodes[v]
                if not v_node.is_alive:
                    continue

                d = u_node.neighbors[v]["distance"]
                etx = u_node.neighbors[v].get("etx", 1.0)
                q_ratio = v_node.queue_occupancy
                e_ratio = v_node.energy.energy_ratio
                v_probs = node_probs.get(v, None)

                cost = self.cost_fn.compute_edge_cost(
                    distance=d,
                    tx_range=network.tx_range,
                    receiver_queue_ratio=q_ratio,
                    receiver_energy_ratio=e_ratio,
                    link_etx=etx,
                    ml_probs=v_probs
                )
                rev_adj[v].append((u, cost))

        pq = [(0.0, sink_id)]
        while pq:
            d_curr, v = heapq.heappop(pq)
            if d_curr > dist[v]:
                continue

            for u, edge_cost in rev_adj[v]:
                new_cost = d_curr + edge_cost
                if new_cost < dist[u]:
                    dist[u] = new_cost
                    next_hop[u] = v
                    heapq.heappush(pq, (new_cost, u))

        for u in range(num_nodes):
            cand_hop = next_hop[u]
            cand_cost = dist[u]

            if cand_hop is not None:
                if self.hysteresis is not None:
                    curr_hop = self.hysteresis.get_active_hop(u)
                    is_curr_alive = (curr_hop is not None and network.nodes[curr_hop].is_alive)
                    if self.hysteresis.should_update_route(u, cand_hop, cand_cost, is_curr_alive):
                        self.routing_table[u] = cand_hop
                        self.path_costs[u] = cand_cost
                    else:
                        self.routing_table[u] = curr_hop
                else:
                    self.routing_table[u] = cand_hop
                    self.path_costs[u] = cand_cost
            else:
                self.routing_table[u] = None

        if self.hysteresis is not None:
            network.total_route_changes = self.hysteresis.route_changes_count

    def get_next_hop(self, u_id: int, network: WSNNetwork) -> Optional[int]:
        if u_id not in self.routing_table:
            self.update_routing_table(network)
        return self.routing_table.get(u_id)
'''

routing_files["V2/routing/astar.py"] = '''import heapq
from typing import Dict, List, Optional, Any, Tuple
from V2.simulator.network import WSNNetwork
from V2.routing.cost_functions import RoutingCostFunction

class AStarRouter:
    def __init__(self, cost_function: RoutingCostFunction, ml_model: Optional[Any] = None, preprocessor: Optional[Any] = None):
        self.cost_fn = cost_function
        self.ml_model = ml_model
        self.preprocessor = preprocessor

    def compute_heuristic(self, node_id: int, network: WSNNetwork) -> float:
        node = network.nodes[node_id]
        norm_dist = node.dist_to_sink / max(1.0, network.tx_range)
        return self.cost_fn.alpha * norm_dist

    def find_path(self, source_id: int, network: WSNNetwork) -> Tuple[Optional[List[int]], int]:
        sink_id = network.sink_id
        if source_id == sink_id:
            return [sink_id], 0

        node_probs = {}
        if self.ml_model is not None:
            features, _ = network.collect_dataset_sample()
            if self.preprocessor is not None:
                features = self.preprocessor.transform(features)
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)
                for i in range(network.num_nodes):
                    node_probs[i] = probs[i]

        open_set = []
        heapq.heappush(open_set, (self.compute_heuristic(source_id, network), 0.0, source_id))
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {i: float("inf") for i in range(network.num_nodes + 1)}
        g_score[source_id] = 0.0
        expansions = 0

        while open_set:
            f, current_g, current = heapq.heappop(open_set)
            expansions += 1
            if current == sink_id:
                path = [sink_id]
                curr = sink_id
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return path, expansions

            if current_g > g_score[current]:
                continue

            curr_node = network.nodes[current]
            for neighbor_id in curr_node.neighbors:
                neighbor_node = network.nodes[neighbor_id]
                if not neighbor_node.is_alive:
                    continue

                d = curr_node.neighbors[neighbor_id]["distance"]
                etx = curr_node.neighbors[neighbor_id].get("etx", 1.0)
                q_ratio = neighbor_node.queue_occupancy
                e_ratio = neighbor_node.energy.energy_ratio
                v_probs = node_probs.get(neighbor_id, None)

                edge_cost = self.cost_fn.compute_edge_cost(
                    distance=d,
                    tx_range=network.tx_range,
                    receiver_queue_ratio=q_ratio,
                    receiver_energy_ratio=e_ratio,
                    link_etx=etx,
                    ml_probs=v_probs
                )
                tentative_g = current_g + edge_cost
                if tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative_g
                    h = self.compute_heuristic(neighbor_id, network)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor_id))

        return None, expansions

    def get_next_hop(self, source_id: int, network: WSNNetwork) -> Optional[int]:
        path, _ = self.find_path(source_id, network)
        if path is not None and len(path) > 1:
            return path[1]
        return None
'''

routing_files["V2/routing/conventional.py"] = '''from typing import Optional
from V2.simulator.network import WSNNetwork
from V2.routing.cost_functions import RoutingCostFunction
from V2.routing.dijkstra import DynamicDijkstraRouter

class ConventionalBaselines:
    @staticmethod
    def get_direct_routing_fn():
        def direct_routing(u_id: int, network: WSNNetwork) -> Optional[int]:
            return network.sink_id
        return direct_routing

    @staticmethod
    def get_min_hop_routing_fn():
        def min_hop_routing(u_id: int, network: WSNNetwork) -> Optional[int]:
            u_node = network.nodes[u_id]
            alive_neighbors = [v for v in u_node.neighbors if network.nodes[v].is_alive]
            if not alive_neighbors:
                return None
            best_v = min(alive_neighbors, key=lambda v: network.nodes[v].hop_count_to_sink)
            return best_v
        return min_hop_routing

    @staticmethod
    def get_pure_distance_router() -> DynamicDijkstraRouter:
        cost_fn = RoutingCostFunction(alpha_dist=1.0, beta_traffic=0.0, gamma_energy=0.0, delta_etx=0.0, eta_ml=0.0)
        return DynamicDijkstraRouter(cost_function=cost_fn)

    @staticmethod
    def get_energy_aware_router() -> DynamicDijkstraRouter:
        cost_fn = RoutingCostFunction(alpha_dist=0.20, beta_traffic=0.0, gamma_energy=0.80, delta_etx=0.0, eta_ml=0.0)
        return DynamicDijkstraRouter(cost_function=cost_fn)

    @staticmethod
    def get_traffic_aware_router() -> DynamicDijkstraRouter:
        cost_fn = RoutingCostFunction(alpha_dist=0.20, beta_traffic=0.80, gamma_energy=0.0, delta_etx=0.0, eta_ml=0.0)
        return DynamicDijkstraRouter(cost_function=cost_fn)

    @staticmethod
    def get_heuristic_multi_metric_router() -> DynamicDijkstraRouter:
        cost_fn = RoutingCostFunction(alpha_dist=0.30, beta_traffic=0.25, gamma_energy=0.25, delta_etx=0.20, eta_ml=0.0)
        return DynamicDijkstraRouter(cost_function=cost_fn)
'''

routing_files["V2/routing/weight_optimizer.py"] = '''import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from V2.simulator.network import WSNNetwork
from V2.routing.cost_functions import RoutingCostFunction
from V2.routing.dijkstra import DynamicDijkstraRouter

class WeightOptimizer:
    def __init__(self, base_config: Dict[str, Any], ml_model: Optional[Any] = None, preprocessor: Optional[Any] = None):
        self.base_config = copy.deepcopy(base_config)
        self.ml_model = ml_model
        self.preprocessor = preprocessor

    def evaluate_weight_vector(self, weights: Tuple[float, float, float, float], sim_duration_s: float = 15.0, dt: float = 0.2) -> Dict[str, Any]:
        alpha, beta, gamma, eta = weights
        cost_fn = RoutingCostFunction(alpha_dist=alpha, beta_traffic=beta, gamma_energy=gamma, delta_etx=0.0, eta_ml=eta)
        router = DynamicDijkstraRouter(cost_function=cost_fn, ml_model=self.ml_model, preprocessor=self.preprocessor)
        cfg = copy.deepcopy(self.base_config)
        cfg["network"]["num_nodes"] = 30
        cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = 3.0
        net = WSNNetwork(cfg)
        steps = int(sim_duration_s / dt)
        reroute_every = int(2.0 / dt)
        for s in range(steps):
            if s % reroute_every == 0:
                router.update_routing_table(net)
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
'''

for path, content in routing_files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Restored: {path}")
