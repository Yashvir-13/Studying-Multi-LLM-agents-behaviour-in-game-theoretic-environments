import argparse
from config import ExperimentConfig
from agents.base_agent import BaseAgent
from agents.llm_client import OllamaClient
from environments.public_goods import PublicsGood
from simulation_runner import SimulationRunner
import json
import os
import metrics


def run_single_simulation(config: ExperimentConfig):
    """Run a single simulation with the given config (uses first value from each list param)."""
    num_agents = config.num_agents[0] if isinstance(config.num_agents, list) else config.num_agents
    multiplier = config.multiplier[0] if isinstance(config.multiplier, list) else config.multiplier
    memory_limit = config.memory_limit[0] if isinstance(config.memory_limit, list) else config.memory_limit
    
    print(f"Starting Single Simulation: N={num_agents}, M={multiplier}, Mem={memory_limit}")
    client = OllamaClient(
        model=config.model,
        temperature=config.temperature,
        num_predict=config.num_predict,
        think=config.think,
    )
    
    agents = [BaseAgent(id=f"Agent_{i}", client=client, memory_limit=memory_limit)
              for i in range(num_agents)]
    
    env = PublicsGood(
        num_of_agents=num_agents,
        initial_endowment=config.initial_endowment,
        multiplier=multiplier
    )
    runner = SimulationRunner(agents, env, steps=config.rounds,
                              observation_type=config.observation_type)
    
    try:
        results = runner.run()
        
        cooperation_rates = metrics.compute_cooperation_rate(results, config.initial_endowment)
        defector_ratios = metrics.compute_defector_ratio(results)
        entropies = metrics.compute_behavior_entropy(results)
        ginis = metrics.compute_reward_gini(results)
        
        final_output = {
            "config": config.to_dict(),
            "metrics": {
                "cooperation_rates": cooperation_rates,
                "defector_ratios": defector_ratios,
                "behavior_entropies": entropies,
                "reward_ginis": ginis
            },
            "history": results
        }

        os.makedirs(config.base_output_dir, exist_ok=True)
        
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(config.base_output_dir, f"single_run_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(final_output, f, indent=4)
        
        latest = os.path.join(config.base_output_dir, "latest_single_run.json")
        with open(latest, 'w') as f:
            json.dump(final_output, f, indent=4)
            
        print(f"\nSingle Run Complete. Saved to {filename}")
        
        if runner.validation_log:
            print(f"\n⚠ {len(runner.validation_log)} out-of-range contributions were clamped:")
            for entry in runner.validation_log[:5]:
                print(f"  Round {entry['round']}, {entry['agent']}: {entry['raw']} → {entry['clamped']}")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()


def run_experiment_sweep(config: ExperimentConfig):
    """Run a parameter sweep using the given config."""
    from experiment_runner import ExperimentRunner
    
    client = OllamaClient(
        model=config.model,
        temperature=config.temperature,
        num_predict=config.num_predict,
        think=config.think,
    )
    runner = ExperimentRunner(client, base_output_dir=config.base_output_dir)
    runner.run_sweep(config)


def run_baseline_sweep(config: ExperimentConfig):
    """Run baseline agent experiments for comparison."""
    from experiment_runner import ExperimentRunner
    from copy import deepcopy
    
    baseline_types = ["always_cooperate", "always_defect", "random", "tit_for_tat"]
    
    runner = ExperimentRunner(client=None, base_output_dir=config.base_output_dir)
    
    for agent_type in baseline_types:
        print(f"\n{'='*60}")
        print(f"Running baseline: {agent_type}")
        print(f"{'='*60}")
        
        baseline_config = deepcopy(config)
        baseline_config.agent_type = agent_type
        baseline_config.memory_limit = [1]  # Memory doesn't matter for most baselines
        baseline_config.trials_per_config = 10
        
        runner.run_sweep(baseline_config)


def main():
    """
    Main entry point for the Multi-Agent LLM Simulation.
    
    Usage:
        python main.py --mode sweep --config public_goods_config.yaml
        python main.py --mode baseline --config public_goods_config.yaml
        python main.py --mode single --config public_goods_config.yaml
        python main.py --mode sweep --config public_goods_config.yaml --resume logs/sweep_xxx
    """
    parser = argparse.ArgumentParser(description="Run Multi-Agent Experiments")
    parser.add_argument("--mode", type=str, default="single", 
                        choices=["single", "sweep", "baseline"],
                        help="Mode: single simulation, parameter sweep, or baseline comparison")
    parser.add_argument("--config", type=str, default="public_goods_config.yaml",
                        help="Path to experiment config file (YAML or JSON)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an interrupted sweep directory to resume (--mode sweep only)")
    args = parser.parse_args()
    
    # Load config
    config = ExperimentConfig.from_file(args.config)
    
    if args.resume:
        config.resume_dir = args.resume
    
    print(f"Loaded config from: {args.config}")
    print(config)
    print()
    
    if args.mode == "sweep":
        run_experiment_sweep(config)
    elif args.mode == "baseline":
        run_baseline_sweep(config)
    else:
        run_single_simulation(config)


if __name__ == "__main__":
    main()
