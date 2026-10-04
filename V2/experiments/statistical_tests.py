"""
Statistical Significance Analysis for WSN Routing Benchmark.
Performs paired t-tests comparing Dueling Double DQN (Deep RL) against conventional and tabular baselines.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats


def run_statistical_significance_suite(
    routing_results_path: str = "V2/results/tables/routing_benchmark_results.csv",
    output_dir: str = "V2/results/tables"
):
    df = pd.read_csv(routing_results_path)
    os.makedirs(output_dir, exist_ok=True)

    comparisons = [
        ("Dueling DDQN (Deep RL)", "Greedy Geographic (GPSR)"),
        ("Dueling DDQN (Deep RL)", "Tabular Q-Routing (RL)"),
        ("Dueling DDQN (Deep RL)", "Minimum Hop (BFS)"),
        ("Dueling DDQN (Deep RL)", "Direct Routing")
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
                if np.std(diff) > 1e-6:
                    t_stat, p_val = stats.ttest_rel(vt, vb)
                else:
                    t_stat, p_val = (0.0, 1.0)

                records.append({
                    "scenario": sc,
                    "comparison": f"{t_m} vs {b_m}",
                    "metric": "PDR (%)",
                    "deep_rl_mean": float(np.mean(vt)),
                    "baseline_mean": float(np.mean(vb)),
                    "mean_difference": float(np.mean(diff)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "statistically_significant": bool(p_val < 0.05)
                })

    df_out = pd.DataFrame(records)
    out_file = os.path.join(output_dir, "statistical_significance_results.csv")
    df_out.to_csv(out_file, index=False)
    print(f"[+] Saved statistical significance results to: {out_file}")
    return df_out


if __name__ == "__main__":
    run_statistical_significance_suite()
