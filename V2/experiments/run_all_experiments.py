from V2.experiments.train_rl_agent import train_rl_routing_agents
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
