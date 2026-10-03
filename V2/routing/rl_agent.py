"""
Deep Reinforcement Learning (Dueling Double DQN) and Tabular Q-Routing for WSN Routing.
Implements:
1. Dueling Double Deep Q-Network (Dueling DDQN) with Action Masking
2. Classical Tabular Q-Routing (Boyan & Littman, NeurIPS 1994)
3. Universal RL Router Wrapper for WSN Discrete-Event Simulator
"""

import collections
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from V2.simulator.network import WSNNetwork
from V2.routing.rl_environment import WSNRoutingRLEnv


# =========================================================================
# 1. Dueling DDQN Neural Network Architecture
# =========================================================================

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 44, action_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared representation encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Value stream: computes scalar state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Advantage stream: computes advantage for each action A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.feature_encoder(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        # Apply action mask: mask out invalid neighbor actions with large negative value
        if action_mask is not None:
            # action_mask has 1 for valid, 0 for invalid
            masked_q = torch.where(action_mask > 0.5, q_values, torch.tensor(-1e9, device=q_values.device))
            return masked_q

        return q_values


# =========================================================================
# 2. Dueling Double DQN Agent with Replay Memory
# =========================================================================

class DuelingDDQNAgent:
    def __init__(
        self,
        state_dim: int = 44,
        action_dim: int = 8,
        gamma: float = 0.95,
        lr: float = 0.0005,
        buffer_size: int = 10000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 150,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update_freq = int(target_update_freq)
        self.device = device
        self.step_counter = 0

        # Online and Target Networks
        self.online_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = collections.deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection respecting the action mask.
        """
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            return 0

        if not evaluate and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        self.online_net.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            m_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(s_tensor, m_tensor)
            action = int(q_values.argmax(dim=-1).item())

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool
    ) -> None:
        self.memory.append((state, action, reward, next_state, next_mask, done))

    def update(self) -> Optional[float]:
        """
        Executes Double DQN Bellman target update from replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)

        s_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        mask_next_t = torch.tensor(np.array(next_masks), dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.online_net.train()
        # Q(s, a)
        current_q = self.online_net(s_t).gather(1, a_t)

        # Double DQN target:
        # Best action selected by ONLINE network: a* = argmax_a Q_online(s', a)
        with torch.no_grad():
            next_q_online = self.online_net(s_next_t, mask_next_t)
            best_next_actions = next_q_online.argmax(dim=-1, keepdim=True)
            # Evaluated by TARGET network: Q_target(s', a*)
            next_q_target = self.target_net(s_next_t, mask_next_t).gather(1, best_next_actions)
            target_q = r_t + (1.0 - done_t) * self.gamma * next_q_target

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay exploration epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, filepath: str) -> None:
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon
        }, filepath)

    def load(self, filepath: str) -> None:
        data = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(data["online_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.epsilon = data.get("epsilon", 0.05)


# =========================================================================
# 3. Classical Tabular Q-Routing (Boyan & Littman, NeurIPS 1994)
# =========================================================================

class TabularQRoutingAgent:
    """
    Each node u maintains estimated Q-values: Q(u, v) = expected reward to route to sink through neighbor v.
    """
    def __init__(self, num_nodes: int = 50, learning_rate: float = 0.1, gamma: float = 0.90, epsilon: float = 0.1):
        self.num_nodes = num_nodes
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: (node_id, neighbor_id) -> Q_value
        self.q_table: Dict[Tuple[int, int], float] = collections.defaultdict(float)

    def select_action(self, current_id: int, valid_neighbors: List[int], evaluate: bool = False) -> Optional[int]:
        if not valid_neighbors:
            return None
        if not evaluate and random.random() < self.epsilon:
            return random.choice(valid_neighbors)

        # Greedily choose neighbor maximizing Q(u, v)
        best_neighbor = max(valid_neighbors, key=lambda v: self.q_table[(current_id, v)])
        return best_neighbor

    def update(self, current_id: int, next_id: int, reward: float, next_valid_neighbors: List[int], done: bool) -> None:
        max_next_q = 0.0
        if not done and next_valid_neighbors:
            max_next_q = max(self.q_table[(next_id, v_next)] for v_next in next_valid_neighbors)

        target = reward + self.gamma * max_next_q
        self.q_table[(current_id, next_id)] += self.alpha * (target - self.q_table[(current_id, next_id)])

    def save(self, filepath: str) -> None:
        import joblib
        joblib.dump(self, filepath)

    def load(self, filepath: str) -> None:
        import joblib
        loaded = joblib.load(filepath)
        if isinstance(loaded, TabularQRoutingAgent):
            self.q_table = loaded.q_table
            self.alpha = loaded.alpha
            self.gamma = loaded.gamma
            self.epsilon = loaded.epsilon
        elif isinstance(loaded, dict):
            self.q_table = collections.defaultdict(float, loaded)


# =========================================================================
# 4. Universal Reinforcement Learning Router Wrappers
# =========================================================================

class DuelingDDQNRouter:
    """
    Adapter allowing Dueling DDQN agent to serve as a routing function for WSNNetwork.step().
    """
    def __init__(self, agent: DuelingDDQNAgent, env: WSNRoutingRLEnv):
        self.agent = agent
        self.env = env
        self.cached_risks: Dict[int, float] = {}

    def refresh_telemetry(self) -> None:
        self.cached_risks = self.env.get_node_ml_risks()

    def get_next_hop(self, u_id: int, network: WSNNetwork) -> Optional[int]:
        state, valid_neighbor_ids, action_mask = self.env.get_state(u_id, self.cached_risks)
        if not valid_neighbor_ids:
            return None

        action_idx = self.agent.select_action(state, action_mask, evaluate=True)
        if action_idx < len(valid_neighbor_ids):
            return valid_neighbor_ids[action_idx]
        return valid_neighbor_ids[0]


class TabularQRouter:
    """
    Adapter allowing Tabular Q-Routing agent to serve as a routing function.
    """
    def __init__(self, agent: TabularQRoutingAgent):
        self.agent = agent

    def get_next_hop(self, u_id: int, network: WSNNetwork) -> Optional[int]:
        u_node = network.nodes[u_id]
        alive_neighbors = [v for v in u_node.neighbors if network.nodes[v].is_alive or v == network.sink_id]
        if not alive_neighbors:
            return None
        return self.agent.select_action(u_id, alive_neighbors, evaluate=True)
