"""
Comprehensive WSN Routing Benchmark Suite.
Compares Conventional Baselines, Tabular Q-Routing, and Deep Reinforcement Learning (Dueling Double DQN)
across factorial traffic regimes (Low, Medium, High, Bursty) and multiple random seeds.
Zero Dijkstra or A* algorithms — strictly decentralized and autonomous routing.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import copy
import joblib
import pandas as pd
import numpy as np
import torch

from V2.simulator.network import WSNNetwork
from V2.routing import (
    ConventionalBaselines, WSNRoutingRLEnv,
    DuelingDDQNAgent, DuelingDDQNRouter, TabularQRoutingAgent, TabularQRouter
)
from V2.models.cnn_1d import TemporalCNN1D
from V2.preprocessing import FeaturePreprocessor


def run_routing_suite(
    config_path: str = "V2/configs/default_config.json",
    output_dir: str = "V2/results",
    sim_duration_s: float = 20.0,
    dt: float = 0.2,
    seeds: list = [42, 43, 44]
):
    print("=" * 80)
    print("   WSN AUTONOMOUS ROUTING BENCHMARK: DEEP RL & CONVENTIONAL BASELINES")
    print("   Algorithms: Direct vs. Min Hop vs. Greedy Geographic vs. Tabular Q vs. Dueling DDQN")
    print("=" * 80)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    with open(config_path, "r") as f:
        base_cfg = json.load(f)

    scaler = FeaturePreprocessor().load("V2/preprocessing/scaler.joblib")

    # Load 1D-CNN model (active telemetry classifier)
    cnn_ckpt = "V2/results/checkpoints/cnn_1d.pt"
    cnn_model = TemporalCNN1D(in_channels=12, seq_len=5, num_classes=3)
    cnn_model.load_state_dict(torch.load(cnn_ckpt, map_location="cpu", weights_only=False))
    cnn_model.eval()

    # Load trained Dueling DDQN agent
    ddqn_ckpt = "V2/results/checkpoints/dueling_ddqn_router.pt"
    ddqn_agent = None
    if os.path.exists(ddqn_ckpt):
        ddqn_agent = DuelingDDQNAgent(state_dim=44, action_dim=8)
        ddqn_agent.load(ddqn_ckpt)
        print("  [+] Loaded pre-trained Dueling DDQN agent successfully.")

    # Load trained Tabular Q agent
    tab_ckpt = "V2/results/checkpoints/tabular_q_router.joblib"
    tabular_q_agent = TabularQRoutingAgent()
    if os.path.exists(tab_ckpt):
        tabular_q_agent.load(tab_ckpt)
        print("  [+] Loaded pre-trained Tabular Q agent successfully.")

    traffic_scenarios = [
        ("Low_CBR", "cbr", 1.5),
        ("Medium_Poisson", "poisson", 3.0),
        ("High_Poisson", "poisson", 6.0),
        ("Bursty_Pareto", "bursty", 4.0)
    ]

    all_records = []

    for traf_name, traf_pat, arr_rate in traffic_scenarios:
        print(f"\n>>> Running Scenario: {traf_name} (Pattern={traf_pat}, Rate={arr_rate} pkts/s) <<<")

        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["network"]["seed"] = seed
            cfg["network"]["num_nodes"] = 35
            cfg["queue_and_traffic"]["traffic_pattern"] = traf_pat
            cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = arr_rate

            # Strictly the 5 active routing methods (No Dijkstra, No A*)
            methods = [
                ("Direct Routing", "Conventional", None, "direct"),
                ("Minimum Hop (BFS)", "Conventional", None, "min_hop"),
                ("Greedy Geographic (GPSR)", "Conventional", ConventionalBaselines.get_greedy_geographic_router(), "greedy"),
                ("Tabular Q-Routing (RL)", "Tabular RL", tabular_q_agent, "tabular"),
                ("Dueling DDQN (Deep RL)", "Deep RL", ddqn_agent, "ddqn")
            ]

            for m_name, m_cat, router_obj, m_type in methods:
                net = WSNNetwork(cfg)
                steps = int(sim_duration_s / dt)
                reroute_steps = int(2.0 / dt)

                start_time = time.perf_counter()

                if m_type == "direct":
                    route_fn = ConventionalBaselines.get_direct_routing_fn()
                    for _ in range(steps):
                        net.step(dt=dt, routing_function=route_fn)
                elif m_type == "min_hop":
                    route_fn = ConventionalBaselines.get_min_hop_routing_fn()
                    for _ in range(steps):
                        net.step(dt=dt, routing_function=route_fn)
                elif m_type == "greedy":
                    for _ in range(steps):
                        net.step(dt=dt, routing_function=router_obj.get_next_hop)
                elif m_type == "tabular":
                    tab_router = TabularQRouter(router_obj)
                    for _ in range(steps):
                        net.step(dt=dt, routing_function=tab_router.get_next_hop)
                elif m_type == "ddqn":
                    rl_env = WSNRoutingRLEnv(net, ml_model=cnn_model, preprocessor=scaler)
                    rl_router = DuelingDDQNRouter(router_obj, rl_env)
                    for s in range(steps):
                        if s % reroute_steps == 0:
                            rl_router.refresh_telemetry()
                        net.step(dt=dt, routing_function=rl_router.get_next_hop)

                elapsed_total = time.perf_counter() - start_time
                metrics = net.compute_metrics()

                rec = {
                    "scenario": traf_name,
                    "traffic_pattern": traf_pat,
                    "arrival_rate": arr_rate,
                    "seed": seed,
                    "method": m_name,
                    "category": m_cat,
                    "pdr_percent": metrics["pdr_percent"],
                    "total_delivered": metrics["total_delivered"],
                    "total_dropped": metrics["total_dropped"],
                    "avg_delay_s": metrics["avg_delay_s"],
                    "p95_delay_s": metrics["p95_delay_s"],
                    "avg_hop_count": metrics["avg_hop_count"],
                    "total_energy_j": metrics["total_energy_j"],
                    "energy_efficiency_bits_per_j": metrics["energy_efficiency_bits_per_j"],
                    "fnd_time_s": metrics["fnd_time_s"],
                    "dead_nodes": metrics["dead_node_count"],
                    "queue_drops": metrics["queue_drops"],
                    "link_drops": metrics["link_drops"],
                    "route_changes": 0,
                    "execution_time_s": elapsed_total
                }
                all_records.append(rec)
                print(f"  [{traf_name}|Seed {seed}] {m_name:<28} -> PDR: {metrics['pdr_percent']:5.1f}% | Delay: {metrics['avg_delay_s']:5.3f}s | Drops: {metrics['total_dropped']:3d}")

    df_results = pd.DataFrame(all_records)
    csv_path = os.path.join(tables_dir, "routing_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[+] Saved detailed routing benchmark results to: {csv_path}")

    # Generate summary table grouped by scenario and method
    summary = df_results.groupby(["scenario", "method", "category"]).agg({
        "pdr_percent": ["mean", "std"],
        "avg_delay_s": ["mean", "std"],
        "total_energy_j": ["mean", "std"],
        "avg_hop_count": "mean",
        "dead_nodes": "mean",
        "route_changes": "mean"
    }).reset_index()

    summary_csv = os.path.join(tables_dir, "routing_summary_by_scenario.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[+] Saved scenario summary to: {summary_csv}")
    return df_results


if __name__ == "__main__":
    run_routing_suite()
