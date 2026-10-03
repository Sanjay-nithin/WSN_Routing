"""
Deep Reinforcement Learning (Dueling Double DQN) Routing Training Pipeline.
Trains an autonomous packet-forwarding agent that learns to avoid congested queues,
depleted batteries, and bad wireless links using trial-and-error experience replay.
"""

import os
import json
import time
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

from V2.simulator.network import WSNNetwork
from V2.simulator.packet import Packet, PacketType
from V2.routing.rl_environment import WSNRoutingRLEnv
from V2.routing.rl_agent import DuelingDDQNAgent, TabularQRoutingAgent
from V2.preprocessing import FeaturePreprocessor


def train_rl_routing_agents(
    config_path: str = "V2/configs/default_config.json",
    output_dir: str = "V2/results",
    num_episodes: int = 120,
    max_steps_per_episode: int = 15
):
    print("=" * 80)
    print("   TRAINING DEEP REINFORCEMENT LEARNING (DUELING DDQN) ROUTING AGENT")
    print("=" * 80)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["network"]["num_nodes"] = 35

    # Load ML classifier for node risk estimation
    scaler = FeaturePreprocessor().load("V2/preprocessing/scaler.joblib")
    rf_model = joblib.load("V2/results/checkpoints/random_forest.joblib")

    network = WSNNetwork(cfg)
    env = WSNRoutingRLEnv(network, ml_model=rf_model, preprocessor=scaler)

    ddqn_agent = DuelingDDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=0.92,
        lr=0.001,
        buffer_size=15000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97
    )

    tabular_q_agent = TabularQRoutingAgent(num_nodes=network.num_nodes + 1)

    episode_rewards = []
    episode_losses = []
    delivered_count = 0
    total_packets = 0

    start_time = time.perf_counter()

    for ep in range(1, num_episodes + 1):
        # Refresh network telemetry occasionally
        if ep % 5 == 0:
            env.network.generate_traffic_step(dt=0.5)

        node_risks = env.get_node_ml_risks()

        # Randomly choose an active source node (excluding sink)
        source_id = np.random.randint(0, network.num_nodes)
        packet = Packet(
            packet_id=ep,
            source_id=source_id,
            dest_id=network.sink_id,
            creation_time=env.network.current_time
        )
        total_packets += 1

        curr_id = source_id
        ep_reward = 0.0
        losses = []

        for step in range(max_steps_per_episode):
            if not packet.add_hop(curr_id):
                break

            state, valid_neighbors, action_mask = env.get_state(curr_id, node_risks)
            if not valid_neighbors:
                break

            action_idx = ddqn_agent.select_action(state, action_mask, evaluate=False)
            if action_idx >= len(valid_neighbors):
                action_idx = 0
            next_id = valid_neighbors[action_idx]

            # Tabular Q agent action
            tab_action = tabular_q_agent.select_action(curr_id, valid_neighbors)

            # Calculate transition reward
            reward = env.calculate_reward(curr_id, next_id, packet, node_risks)
            ep_reward += reward

            done = (next_id == network.sink_id)

            if done:
                packet.mark_delivered(env.network.current_time)
                delivered_count += 1
                next_state = np.zeros_like(state)
                next_mask = np.zeros_like(action_mask)
            else:
                next_state, next_valid, next_mask = env.get_state(next_id, node_risks)

            # Store in DDQN replay buffer
            ddqn_agent.store_transition(state, action_idx, reward, next_state, next_mask, done)

            # Tabular Q update
            next_val_list = [v for v in network.nodes[next_id].neighbors if network.nodes[v].is_alive or v == network.sink_id] if not done else []
            tabular_q_agent.update(curr_id, tab_action, reward, next_val_list, done)

            # Train DDQN
            loss = ddqn_agent.update()
            if loss is not None:
                losses.append(loss)

            if done:
                break

            curr_id = next_id

        episode_rewards.append(ep_reward)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        episode_losses.append(avg_loss)

        if ep % 20 == 0 or ep == num_episodes:
            success_rate = (delivered_count / total_packets) * 100.0
            print(f"  Episode {ep:3d}/{num_episodes} | Avg Reward: {np.mean(episode_rewards[-20:]):6.2f} | Loss: {avg_loss:.4f} | Epsilon: {ddqn_agent.epsilon:.3f} | Success Rate: {success_rate:.1f}%")

    elapsed = time.perf_counter() - start_time
    print(f"\n[+] RL Training completed in {elapsed:.1f} seconds!")

    # Save models
    ddqn_path = os.path.join(ckpt_dir, "dueling_ddqn_router.pt")
    ddqn_agent.save(ddqn_path)
    joblib.dump(tabular_q_agent, os.path.join(ckpt_dir, "tabular_q_router.joblib"))
    print(f"[+] Saved trained Dueling DDQN model to: {ddqn_path}")

    # Plot RL learning curves
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    # Moving average
    window = 10
    ma_rewards = [np.mean(episode_rewards[max(0, i-window):i+1]) for i in range(len(episode_rewards))]
    plt.plot(episode_rewards, alpha=0.3, color="teal", label="Raw Reward")
    plt.plot(ma_rewards, color="teal", linewidth=2.0, label="10-Ep Moving Avg")
    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Dueling DDQN: Episode Reward Progression")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, color="crimson", linewidth=1.5)
    plt.xlabel("Training Episode")
    plt.ylabel("Smooth L1 TD-Loss")
    plt.title("Dueling DDQN: Temporal Difference Loss")
    plt.tight_layout()

    curve_path = os.path.join(plots_dir, "fig8_rl_training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[+] Saved RL training curves to: {curve_path}")

    return ddqn_agent, tabular_q_agent


if __name__ == "__main__":
    train_rl_routing_agents()
