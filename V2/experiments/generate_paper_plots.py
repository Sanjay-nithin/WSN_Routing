import os
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
