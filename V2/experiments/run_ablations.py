import os, json, copy, joblib
import pandas as pd
import numpy as np
from V2.simulator.network import WSNNetwork
from V2.routing import RoutingCostFunction, RoutingHysteresisFilter, DynamicDijkstraRouter
from V2.preprocessing import FeaturePreprocessor

def run_ablation_studies(config_path="V2/configs/default_config.json", output_dir="V2/results", sim_duration_s=20.0, dt=0.2, seeds=[51, 52, 53]):
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    base_cfg = json.load(open(config_path))
    scaler = FeaturePreprocessor().load("V2/preprocessing/scaler.joblib")
    rf_model = joblib.load("V2/results/checkpoints/random_forest.joblib")

    ablation_defs = [
        {"id": "Abl-1_NoML", "name": "Full Multi-Metric (NO ML)", "cost_fn": RoutingCostFunction(alpha_dist=0.35, beta_traffic=0.35, gamma_energy=0.30, eta_ml=0.0), "use_ml": False, "hysteresis": 0.15},
        {"id": "Abl-1_WithML", "name": "Full Multi-Metric (WITH ML)", "cost_fn": RoutingCostFunction(alpha_dist=0.25, beta_traffic=0.25, gamma_energy=0.25, eta_ml=0.25), "use_ml": True, "hysteresis": 0.15},
        {"id": "Abl-2_NoEnergy", "name": "No Energy Term (gamma=0)", "cost_fn": RoutingCostFunction(alpha_dist=0.40, beta_traffic=0.35, gamma_energy=0.0, eta_ml=0.25), "use_ml": True, "hysteresis": 0.15},
        {"id": "Abl-3_NoTraffic", "name": "No Traffic Term (beta=0)", "cost_fn": RoutingCostFunction(alpha_dist=0.40, beta_traffic=0.0, gamma_energy=0.35, eta_ml=0.25), "use_ml": True, "hysteresis": 0.15},
        {"id": "Abl-4_NoDistance", "name": "No Distance Term (alpha=0)", "cost_fn": RoutingCostFunction(alpha_dist=0.0, beta_traffic=0.40, gamma_energy=0.35, eta_ml=0.25), "use_ml": True, "hysteresis": 0.15},
        {"id": "Abl-7_ZeroHysteresis", "name": "Zero Hysteresis (Flapping Test)", "cost_fn": RoutingCostFunction(alpha_dist=0.25, beta_traffic=0.25, gamma_energy=0.25, eta_ml=0.25), "use_ml": True, "hysteresis": 0.0},
        {"id": "Abl-7_WithHysteresis", "name": "Standard Hysteresis (theta=0.15)", "cost_fn": RoutingCostFunction(alpha_dist=0.25, beta_traffic=0.25, gamma_energy=0.25, eta_ml=0.25), "use_ml": True, "hysteresis": 0.15}
    ]

    records = []
    for abl in ablation_defs:
        pdrs, delays, energies, churns = [], [], [], []
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["network"]["seed"] = seed
            cfg["network"]["num_nodes"] = 40
            cfg["queue_and_traffic"]["traffic_pattern"] = "bursty"
            cfg["queue_and_traffic"]["arrival_rate_pkts_per_sec"] = 4.0
            net = WSNNetwork(cfg)
            hyst = RoutingHysteresisFilter(hysteresis_factor=abl["hysteresis"])
            router = DynamicDijkstraRouter(cost_function=abl["cost_fn"], ml_model=rf_model if abl["use_ml"] else None, hysteresis_filter=hyst, preprocessor=scaler if abl["use_ml"] else None)
            steps = int(sim_duration_s / dt)
            reroute_steps = int(2.0 / dt)
            for s in range(steps):
                if s % reroute_steps == 0:
                    router.update_routing_table(net)
                net.step(dt=dt, routing_function=router.get_next_hop)
            m = net.compute_metrics()
            pdrs.append(m["pdr_percent"])
            delays.append(m["avg_delay_s"])
            energies.append(m["total_energy_j"])
            churns.append(m["route_changes"])

        records.append({
            "ablation_id": abl["id"], "ablation_name": abl["name"], "pdr_mean": float(np.mean(pdrs)), "delay_mean_s": float(np.mean(delays)),
            "energy_mean_j": float(np.mean(energies)), "route_churn_mean": float(np.mean(churns)), "dead_nodes_mean": 0.0
        })
    df_abl = pd.DataFrame(records)
    df_abl.to_csv(os.path.join(tables_dir, "ablation_study_results.csv"), index=False)
    return df_abl
