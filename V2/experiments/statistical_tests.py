import os
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
