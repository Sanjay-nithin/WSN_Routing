exp_files = {}

exp_files["V2/experiments/run_ablations.py"] = '''import os, json, copy, joblib
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
'''

exp_files["V2/experiments/run_sensitivity.py"] = '''import os, json, joblib
import pandas as pd
from V2.routing.weight_optimizer import WeightOptimizer
from V2.preprocessing import FeaturePreprocessor

def run_sensitivity_analysis(config_path="V2/configs/default_config.json", output_dir="V2/results", num_samples=30):
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    base_cfg = json.load(open(config_path))
    scaler = FeaturePreprocessor().load("V2/preprocessing/scaler.joblib")
    rf_model = joblib.load("V2/results/checkpoints/random_forest.joblib")
    optimizer = WeightOptimizer(base_config=base_cfg, ml_model=rf_model, preprocessor=scaler)
    results = optimizer.run_dirichlet_search(num_samples=num_samples, seed=42)
    df_sens = pd.DataFrame(results)
    df_sens.to_csv(os.path.join(tables_dir, "weight_sensitivity_results.csv"), index=False)
    pareto_results = optimizer.filter_pareto_optimal(results)
    df_pareto = pd.DataFrame(pareto_results)
    df_pareto.to_csv(os.path.join(tables_dir, "pareto_optimal_weights.csv"), index=False)
    return df_pareto
'''

exp_files["V2/experiments/statistical_tests.py"] = '''import os
import pandas as pd
import numpy as np
from scipy import stats

def run_statistical_significance_suite(routing_results_path="V2/results/tables/routing_benchmark_results.csv", output_dir="V2/results/tables"):
    df = pd.read_csv(routing_results_path)
    os.makedirs(output_dir, exist_ok=True)
    comparisons = [
        ("RF + A* Search", "Distance Dijkstra"),
        ("RF + Dynamic Dijkstra", "Distance Dijkstra"),
        ("Dueling DDQN (Deep RL)", "Distance Dijkstra")
    ]
    records = []
    for sc in df["scenario"].unique():
        df_sc = df[df["scenario"] == sc]
        for t_m, b_m in comparisons:
            sub_t = df_sc[df_sc["method"] == t_m]
            sub_b = df_sc[df_sc["method"] == b_m]
            if len(sub_t) >= 2 and len(sub_b) >= 2:
                vt = sub_t["pdr_percent"].values
                vb = sub_b["pdr_percent"].values
                diff = vt - vb
                t_stat, p_val = stats.ttest_rel(vt, vb) if np.std(diff) > 1e-6 else (0.0, 1.0)
                records.append({
                    "scenario": sc, "comparison": f"{t_m} vs {b_m}", "mean_diff": float(np.mean(diff)),
                    "p_value": float(p_val), "significant": bool(p_val < 0.05)
                })
    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(output_dir, "statistical_significance_results.csv"), index=False)
    return df_out
'''

exp_files["V2/experiments/generate_paper_plots.py"] = '''import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_plots(results_dir="V2/results"):
    tables_dir = os.path.join(results_dir, "tables")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300})

    rout_path = os.path.join(tables_dir, "routing_benchmark_results.csv")
    if os.path.exists(rout_path):
        df_rout = pd.read_csv(rout_path)
        plt.figure(figsize=(11, 6))
        # Top representative methods
        repr_m = ["Direct Routing", "Distance Dijkstra", "Heuristic Multi-Metric", "RF + A* Search", "Dueling DDQN (Deep RL)"]
        sub = df_rout[df_rout["method"].isin(repr_m)]
        sns.barplot(data=sub, x="scenario", y="pdr_percent", hue="method")
        plt.xlabel("Traffic Regime")
        plt.ylabel("Packet Delivery Ratio (PDR %)")
        plt.title("Figure 2: Routing Delivery (PDR) across Traffic Regimes (Including Deep RL)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig2_pdr_comparison_by_traffic.png"))
        plt.close()
'''

exp_files["V2/experiments/run_all_experiments.py"] = '''from V2.experiments.train_rl_agent import train_rl_routing_agents
from V2.experiments.run_routing_suite import run_routing_suite
from V2.experiments.generate_paper_plots import generate_all_plots

def main():
    print("Training RL agent...")
    train_rl_routing_agents(num_episodes=100)
    print("Running Routing Suite with RL...")
    run_routing_suite()
    print("Generating Plots...")
    generate_all_plots()

if __name__ == "__main__":
    main()
'''

for path, content in exp_files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Restored: {path}")
