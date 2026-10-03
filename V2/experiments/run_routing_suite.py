"""
Comprehensive WSN Routing Benchmark Suite.
Compares Conventional Baselines, Heuristics, and Deep Reinforcement Learning (Dueling DDQN)
across factorial traffic regimes (Low, Medium, High, Bursty) and multiple random seeds.
"""

import os
import json
import time
import copy
import joblib
import pandas as pd
import numpy as np

from V2.simulator.network import WSNNetwork
from V2.routing import (
    RoutingCostFunction, RoutingHysteresisFilter, DynamicDijkstraRouter,
    AStarRouter, ConventionalBaselines, WSNRoutingRLEnv,
    DuelingDDQNAgent, DuelingDDQNRouter, TabularQRoutingAgent, TabularQRouter
)
from V2.preprocessing import FeaturePreprocessor


def run_routing_suite(
    config_path: str = "V2/configs/default_config.json",
    output_dir: str = "V2/results",
    sim_duration_s: float = 20.0,
    dt: float = 0.2,
    seeds: list = [42, 43, 44]
):
    print("=" * 80)
    print("   WSN DYNAMIC ROUTING BENCHMARK: REINFORCEMENT LEARNING & BASELINES")
    print("=" * 80)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    with open(config_path, "r") as f:
        base_cfg = json.load(f)

    scaler = FeaturePreprocessor().load("V2/preprocessing/scaler.joblib")
    rf_model = joblib.load("V2/results/checkpoints/random_forest.joblib")

    # Load trained Dueling DDQN agent
    ddqn_ckpt = "V2/results/checkpoints/dueling_ddqn_router.pt"
    ddqn_agent = None
    if os.path.exists(ddqn_ckpt):
        ddqn_agent = DuelingDDQNAgent(state_dim=44, action_dim=8)
        ddqn_agent.load(ddqn_ckpt)
        print("  [+] Loaded pre-trained Dueling DDQN agent successfully.")

    # Load trained Tabular Q agent
    tab_ckpt = "V2/results/checkpoints/tabular_q_router.joblib"
    tabular_q_agent = joblib.load(tab_ckpt) if os.path.exists(tab_ckpt) else None

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
            cfg["network"]["num_nodes"] = 40
            cfg["queue_and_traffic"]["traffic_pattern"] = traf_pat
            cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = arr_rate

            # Base routing methods
            methods = [
                ("Direct Routing", "Conventional", None, "direct"),
                ("Minimum Hop (BFS)", "Conventional", None, "min_hop"),
                ("Distance Dijkstra", "Conventional", ConventionalBaselines.get_pure_distance_router(), "dijkstra"),
                ("Energy-Aware Dijkstra", "Heuristic", ConventionalBaselines.get_energy_aware_router(), "dijkstra"),
                ("Traffic-Aware Dijkstra", "Heuristic", ConventionalBaselines.get_traffic_aware_router(), "dijkstra"),
                ("Heuristic Multi-Metric", "Heuristic", ConventionalBaselines.get_heuristic_multi_metric_router(), "dijkstra"),
                ("RF + Dynamic Dijkstra", "ML-Heuristic", DynamicDijkstraRouter(
                    cost_function=RoutingCostFunction(alpha_dist=0.25, beta_traffic=0.20, gamma_energy=0.20, eta_ml=0.35),
                    ml_model=rf_model,
                    hysteresis_filter=RoutingHysteresisFilter(0.15),
                    preprocessor=scaler
                ), "dijkstra"),
                ("RF + A* Search", "ML-Informed", AStarRouter(
                    cost_function=RoutingCostFunction(alpha_dist=0.25, beta_traffic=0.20, gamma_energy=0.20, eta_ml=0.35),
                    ml_model=rf_model,
                    preprocessor=scaler
                ), "astar")
            ]

            # Add Reinforcement Learning methods
            if ddqn_agent is not None:
                methods.append(("Dueling DDQN (Deep RL)", "Deep RL", "ddqn", "rl"))
            if tabular_q_agent is not None:
                methods.append(("Tabular Q-Routing (RL)", "Tabular RL", "tabular", "rl"))

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
                elif m_type == "dijkstra":
                    router_obj.update_routing_table(net)
                    for s in range(steps):
                        if s > 0 and s % reroute_steps == 0:
                            router_obj.update_routing_table(net)
                        net.step(dt=dt, routing_function=router_obj.get_next_hop)
                elif m_type == "astar":
                    for _ in range(steps):
                        net.step(dt=dt, routing_function=router_obj.get_next_hop)
                elif m_type == "rl":
                    if router_obj == "ddqn":
                        rl_env = WSNRoutingRLEnv(net, ml_model=rf_model, preprocessor=scaler)
                        rl_router = DuelingDDQNRouter(ddqn_agent, rl_env)
                        for s in range(steps):
                            if s % reroute_steps == 0:
                                rl_router.refresh_telemetry()
                            net.step(dt=dt, routing_function=rl_router.get_next_hop)
                    elif router_obj == "tabular":
                        tab_router = TabularQRouter(tabular_q_agent)
                        for _ in range(steps):
                            net.step(dt=dt, routing_function=tab_router.get_next_hop)

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
                    "route_changes": metrics["route_changes"],
                    "execution_time_s": elapsed_total
                }
                all_records.append(rec)
                print(f"  [{m_name:28s}] Seed={seed} | PDR: {rec['pdr_percent']:5.1f}% | Delay: {rec['avg_delay_s']:6.3f}s | Energy: {rec['total_energy_j']:6.2f}J")

    df_routing = pd.DataFrame(all_records)
    csv_out = os.path.join(tables_dir, "routing_benchmark_results.csv")
    df_routing.to_csv(csv_out, index=False)

    agg_df = df_routing.groupby(["scenario", "method", "category"]).agg({
        "pdr_percent": ["mean", "std"],
        "avg_delay_s": ["mean", "std"],
        "total_energy_j": ["mean", "std"],
        "avg_hop_count": ["mean"],
        "dead_nodes": ["mean"]
    }).reset_index()

    agg_csv = os.path.join(tables_dir, "routing_summary_by_scenario.csv")
    agg_df.to_csv(agg_csv, index=False)

    print("\n" + "=" * 80)
    print("      AGGREGATED ROUTING PERFORMANCE TABLE WITH RL INCLUDED")
    print("=" * 80)
    print(agg_df.to_string())
    print(f"\n[+] Raw results saved to: {csv_out}")
    print(f"[+] Aggregated summary saved to: {agg_csv}")
    return df_routing


if __name__ == "__main__":
    run_routing_suite()
