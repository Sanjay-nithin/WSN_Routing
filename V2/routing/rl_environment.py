"""
Reinforcement Learning Environment Wrapper for Wireless Sensor Network Routing.
Formulates packet forwarding as a Markov Decision Process (MDP):
- State Space: Current node telemetry + Candidate 1-hop neighbors' states + ML Health Risk
- Action Space: Discrete next-hop forwarding decisions among active neighbors
- Reward: Multi-objective formulation penalizing delay, queue bloat, energy depletion, and routing loops.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from V2.simulator.network import WSNNetwork
from V2.simulator.packet import Packet, DropReason


class WSNRoutingRLEnv:
    def __init__(
        self,
        network: WSNNetwork,
        ml_model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        max_neighbors: int = 8
    ):
        self.network = network
        self.ml_model = ml_model
        self.preprocessor = preprocessor
        self.max_neighbors = max_neighbors

        # State representation dimension:
        # 4 local node features + (max_neighbors * 5 neighbor features)
        # Local: [E_res, Q_occ, Dist_BS, Degree]
        # Per Neighbor: [Valid_flag, Rel_dist_BS, Q_occ, E_res, ML_Risk]
        self.state_dim = 4 + (self.max_neighbors * 5)
        self.action_dim = self.max_neighbors

    def get_node_ml_risks(self) -> Dict[int, float]:
        """
        Extracts continuous ML health risk for each node using the pre-trained classifier.
        """
        risks = {}
        if self.ml_model is not None:
            features, _ = self.network.collect_dataset_sample()
            if self.preprocessor is not None:
                features = self.preprocessor.transform(features)
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)
                for i in range(self.network.num_nodes):
                    # Risk: w_c * P(Congested) + w_u * P(Unhealthy)
                    p_c = probs[i][1]
                    p_u = probs[i][2]
                    risks[i] = float(np.clip((0.8 * p_c + 2.0 * p_u) / 2.0, 0.0, 1.0))
        for i in range(self.network.num_nodes + 1):
            if i not in risks:
                risks[i] = 0.0
        return risks

    def get_state(self, current_node_id: int, node_risks: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """
        Constructs the state vector and action mask for the current node.
        Returns:
            state_vec: np.ndarray of shape (state_dim,)
            valid_neighbor_ids: list of actual node IDs corresponding to discrete actions
            action_mask: np.ndarray of shape (action_dim,) where 1 = valid neighbor, 0 = invalid/masked
        """
        risks = node_risks if node_risks is not None else self.get_node_ml_risks()
        u_node = self.network.nodes[current_node_id]

        # 1. Local node telemetry
        e_res = u_node.energy.energy_ratio if not u_node.is_sink else 1.0
        q_occ = u_node.queue_occupancy
        d_bs = u_node.dist_to_sink / max(1.0, self.network.max_field_dim)
        deg = len(u_node.neighbors) / max(1.0, self.network.num_nodes)

        local_feats = [e_res, q_occ, d_bs, deg]

        # 2. Neighbor candidate features
        # Filter candidate neighbors to forwarding cone (closer to sink or within margin) to prevent ping-pong loops
        alive_neighbors = [
            v for v in u_node.neighbors
            if (self.network.nodes[v].is_alive or v == self.network.sink_id) and
               (self.network.nodes[v].dist_to_sink <= u_node.dist_to_sink + 0.3 * self.network.tx_range)
        ]
        if not alive_neighbors:
            alive_neighbors = [
                v for v in u_node.neighbors if self.network.nodes[v].is_alive or v == self.network.sink_id
            ]
        alive_neighbors.sort(key=lambda v: self.network.nodes[v].dist_to_sink)

        neighbor_feats = []
        valid_neighbor_ids = []
        action_mask = np.zeros(self.max_neighbors, dtype=np.float32)

        for i in range(self.max_neighbors):
            if i < len(alive_neighbors):
                v_id = alive_neighbors[i]
                v_node = self.network.nodes[v_id]
                valid_neighbor_ids.append(v_id)
                action_mask[i] = 1.0

                rel_d_bs = (v_node.dist_to_sink - u_node.dist_to_sink) / max(1.0, self.network.tx_range)
                v_q = v_node.queue_occupancy if not v_node.is_sink else 0.0
                v_e = v_node.energy.energy_ratio if not v_node.is_sink else 1.0
                v_risk = risks.get(v_id, 0.0)

                neighbor_feats.extend([1.0, rel_d_bs, v_q, v_e, v_risk])
            else:
                neighbor_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        state_vec = np.array(local_feats + neighbor_feats, dtype=np.float32)
        return state_vec, valid_neighbor_ids, action_mask

    def calculate_reward(
        self,
        current_id: int,
        next_id: int,
        packet: Packet,
        node_risks: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Multi-objective reinforcement learning reward:
        - Big positive reward for delivery to Base Station (+10.0)
        - Small progress reward for moving closer to Base Station
        - Negative penalties for:
            * High buffer occupancy at next hop (-2.0 * Q_occ)
            * Depleted battery at next hop (-2.0 * (1 - E_res))
            * High ML health/congestion risk (-3.0 * Risk)
            * Routing loop / revisiting nodes (-8.0)
        """
        risks = node_risks if node_risks is not None else self.get_node_ml_risks()
        sink_id = self.network.sink_id

        # 1. Terminal Goal: Delivery to Base Station
        if next_id == sink_id:
            return 10.0

        u_node = self.network.nodes[current_id]
        v_node = self.network.nodes[next_id]

        # 2. Check for Routing Loops (packet revisiting previous node)
        if next_id in packet.path:
            return -8.0

        # 3. Geometric Progress toward Sink
        progress = (u_node.dist_to_sink - v_node.dist_to_sink) / max(1.0, self.network.tx_range)
        progress_reward = 1.5 * progress

        # 4. Congestion & Buffer Bloat Penalty
        queue_penalty = -2.5 * v_node.queue_occupancy

        # 5. Battery Depletion Penalty
        energy_penalty = -2.0 * (1.0 - v_node.energy.energy_ratio)

        # 6. ML Health Risk Penalty
        risk_penalty = -3.5 * risks.get(next_id, 0.0)

        # Baseline per-hop step cost
        step_cost = -0.2

        total_reward = progress_reward + queue_penalty + energy_penalty + risk_penalty + step_cost
        return float(total_reward)
