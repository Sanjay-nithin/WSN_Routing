import os, json, joblib
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
