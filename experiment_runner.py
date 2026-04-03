import itertools
import numpy as np
import json
import os
import time
from tqdm import tqdm
from agents.base_agent import BaseAgent
from agents.baseline_agents import AlwaysCooperateAgent, AlwaysDefectAgent, RandomAgent, TitForTatAgent
from agents.signaling_agents import SenderAgent, ReceiverAgent
from agents.signaling_baseline_agents import SENDER_BASELINE_TYPES, RECEIVER_BASELINE_TYPES
from environments.public_goods import PublicsGood
from simulation_runner import SimulationRunner
import metrics

# Map agent type strings to classes
AGENT_TYPES = {
    "llm": None,  # Uses BaseAgent with LLM client
    "always_cooperate": AlwaysCooperateAgent,
    "always_defect": AlwaysDefectAgent,
    "random": RandomAgent,
    "tit_for_tat": TitForTatAgent,
}


class ExperimentRunner:
    """
    Orchestrates parameter sweeps for multi-agent experiments.
    
    Accepts an ExperimentConfig (or legacy dict) and iterates through 
    all parameter combinations, running multiple trials per config.
    Auto-saves the full config to each output directory for reproducibility.
    """
    def __init__(self, client, base_output_dir="logs"):
        self.client = client
        self.base_output_dir = base_output_dir

    def _config_key(self, config):
        """Generates a unique string key for a parameter configuration."""
        agent_type = config.get('agent_type', 'llm')
        env = config.get('environment', 'public_good')
        
        if env == 'signaling_game':
            action_mode = config.get("receiver_action_mode", "global")
            reveal_flag = int(bool(config.get("reveal_sender_states", True)))
            base = (
                f"S{config['num_senders']}_R{config['num_receivers']}"
                f"_T{config['topology']}_Mem{config['memory_limit']}"
                f"_A{action_mode}_F{reveal_flag}"
            )
        else:
            base = f"A{config['num_agents']}_M{config['multiplier']}_Mem{config['memory_limit']}"
        
        if agent_type != "llm":
            base += f"_{agent_type}"
        return base

    def _find_completed_configs(self, output_dir):
        """Scans for existing raw_*.json files to detect completed configs."""
        completed = set()
        if not os.path.exists(output_dir):
            return completed
        for filename in os.listdir(output_dir):
            if filename.startswith("raw_") and filename.endswith(".json"):
                key = filename[len("raw_"):-len(".json")]
                completed.add(key)
        return completed

    def run_sweep(self, experiment_config):
        """
        Runs a full parameter sweep.
        
        Args:
            experiment_config: An ExperimentConfig object or legacy dict.
                If ExperimentConfig, uses its attributes directly.
                If dict, uses it as the old sweep_config format.
        
        Returns:
            list: Aggregated results for all configurations.
        """
        # Support both ExperimentConfig and legacy dict
        from config import ExperimentConfig
        if isinstance(experiment_config, ExperimentConfig):
            sweep_config = experiment_config.get_sweep_config()
            resume_dir = experiment_config.resume_dir
        else:
            sweep_config = experiment_config
            resume_dir = sweep_config.pop("resume_dir", None)
        
        # Generate all combinations based on environment type
        env_type = sweep_config.get("environment", "public_good")
        
        if env_type == "signaling_game":
            keys = ["num_senders", "num_receivers", "topology", "memory_limit"]
        else:
            keys = ["num_agents", "multiplier", "memory_limit"]
        
        values = [sweep_config[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        # Determine output directory (new or resumed)
        if resume_dir and os.path.isdir(resume_dir):
            output_dir = resume_dir
            print(f"Resuming sweep in: {output_dir}")
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            agent_type = sweep_config.get("agent_type", "llm")
            prefix = f"baseline_{agent_type}" if agent_type != "llm" else "sweep"
            experiment_id = f"{prefix}_{timestamp}"
            output_dir = os.path.join(self.base_output_dir, experiment_id)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Starting new sweep in: {output_dir}")
        
        # AUTO-SAVE: Save full config for reproducibility
        if isinstance(experiment_config, ExperimentConfig):
            experiment_config.save(output_dir)
        else:
            config_path = os.path.join(output_dir, "sweep_config.json")
            with open(config_path, "w") as f:
                json.dump(sweep_config, f, indent=4)
        
        # Detect completed configs
        completed_keys = self._find_completed_configs(output_dir)
        if completed_keys:
            print(f"Found {len(completed_keys)} completed config(s): {completed_keys}")
        
        # Load existing partial results
        report_path = os.path.join(output_dir, "final_sweep_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                aggregated_results = json.load(f)
        else:
            aggregated_results = []

        total = len(combinations)
        skipped = 0
        
        config_pbar = tqdm(enumerate(combinations), total=total, desc="Configs", unit="cfg")

        for i, combo in config_pbar:
            config = dict(zip(keys, combo))
            config["environment"] = env_type
            config["rounds"] = sweep_config["rounds"]
            config["trials"] = sweep_config["trials_per_config"]
            config["agent_type"] = sweep_config.get("agent_type", "llm")
            
            if env_type == "signaling_game":
                config["high_probability"] = sweep_config.get("high_probability", 0.5)
                config["receiver_action_mode"] = sweep_config.get("receiver_action_mode", "global")
                config["reveal_sender_states"] = sweep_config.get("reveal_sender_states", True)
                config["sender_reward_high_invest"] = sweep_config.get("sender_reward_high_invest", sweep_config.get("sender_reward_invest", 3.0))
                config["sender_reward_low_invest"] = sweep_config.get("sender_reward_low_invest", sweep_config.get("sender_reward_invest", 3.0))
                config["sender_reward_invest"] = sweep_config.get("sender_reward_invest", 3.0)
                config["sender_reward_pass"] = sweep_config.get("sender_reward_pass", 0.0)
                config["receiver_reward_high_invest"] = sweep_config.get("receiver_reward_high_invest", 3.0)
                config["receiver_reward_low_invest"] = sweep_config.get("receiver_reward_low_invest", -1.0)
                config["receiver_reward_pass"] = sweep_config.get("receiver_reward_pass", 0.0)
            else:
                config["initial_endowment"] = sweep_config.get("initial_endowment", 10)
                config["observation_type"] = sweep_config.get("observation_type", "full")
            
            config_key = self._config_key(config)
            
            if config_key in completed_keys:
                skipped += 1
                config_pbar.set_postfix_str(f"SKIP {config_key}")
                continue
            
            config_pbar.set_postfix_str(config_key)
            
            config_metrics = self.run_experiment(config, output_dir)
            aggregated_results.append({
                "config": config,
                "metrics": config_metrics
            })
            
            # CHECKPOINT
            with open(report_path, "w") as f:
                json.dump(aggregated_results, f, indent=4)
            print(f"  [CHECKPOINT] Saved progress ({len(aggregated_results)} configs done)")
            
        print(f"\nSweep Complete! {total - skipped} new configs run, {skipped} skipped (resumed).")
        print(f"Final report: {report_path}")
            
        return aggregated_results

    def run_experiment(self, config, output_dir):
        """Runs N trials for a single configuration and saves raw data."""
        trials_data = []
        config_key = self._config_key(config)
        agent_type = config.get("agent_type", "llm")
        env_type = config.get("environment", "public_good")
        
        for t in tqdm(range(config["trials"]), desc=f"  Trials ({config_key})", unit="trial", leave=False):
            
            if env_type == "signaling_game":
                history = self._run_signaling_trial(config, agent_type)
            else:
                history = self._run_public_goods_trial(config, agent_type)
            
            # Compute metrics based on environment
            if env_type == "signaling_game":
                trial_metrics = self._compute_signaling_metrics(history)
            else:
                trial_metrics = self._compute_public_goods_metrics(history, config)
            
            trials_data.append({
                "trial_id": t,
                "history": history,
                "metrics": trial_metrics
            })
        
        # Aggregate and save
        if env_type == "signaling_game":
            aggregated = self._aggregate_signaling_metrics(trials_data, config)
        else:
            aggregated = self._aggregate_public_goods_metrics(trials_data, config)
        
        with open(os.path.join(output_dir, f"raw_{config_key}.json"), "w") as f:
            json.dump(trials_data, f, indent=4)
        
        return aggregated
    
    # ================================
    # Public Goods Game helpers
    # ================================
    
    def _run_public_goods_trial(self, config, agent_type):
        """Run a single trial of the Public Goods Game."""
        endowment = config.get("initial_endowment", 10)
        obs_type = config.get("observation_type", "full")
        
        agents = []
        for i in range(config["num_agents"]):
            if agent_type == "llm":
                agent = BaseAgent(id=f"Agent_{i}", client=self.client, memory_limit=config["memory_limit"])
            else:
                agent_class = AGENT_TYPES.get(agent_type)
                if agent_class is None:
                    raise ValueError(f"Unknown agent_type: {agent_type}")
                agent = agent_class(id=f"Agent_{i}", memory_limit=config["memory_limit"])
            agents.append(agent)
        
        env = PublicsGood(
            num_of_agents=config["num_agents"],
            initial_endowment=endowment,
            multiplier=config["multiplier"]
        )
        
        runner = SimulationRunner(agents, env, steps=config["rounds"], observation_type=obs_type)
        return runner.run()
    
    def _compute_public_goods_metrics(self, history, config):
        """Compute all 7 public goods metrics for a single trial."""
        endowment = config.get("initial_endowment", 10)
        coop = metrics.compute_cooperation_rate(history, endowment)
        return {
            "cooperation_rates": coop,
            "entropy": metrics.compute_behavior_entropy(history),
            "gini": metrics.compute_reward_gini(history),
            "defector_ratio": metrics.compute_defector_ratio(history),
            "stability_index": metrics.compute_stability_index(coop),
            "time_to_convergence": metrics.compute_time_to_convergence(coop),
            "change_points": metrics.detect_change_point(coop),
        }
    
    def _aggregate_public_goods_metrics(self, trials_data, config):
        """Aggregate public goods metrics across trials."""
        rounds = config["rounds"]
        n_trials = config["trials"]
        
        coop_matrix = np.zeros((n_trials, rounds))
        entropy_matrix = np.zeros((n_trials, rounds))
        gini_matrix = np.zeros((n_trials, rounds))
        defector_matrix = np.zeros((n_trials, rounds))
        convergence_times = []
        all_change_points = []
        
        for t, data in enumerate(trials_data):
            m = data["metrics"]
            coop_matrix[t, :len(m["cooperation_rates"])] = m["cooperation_rates"]
            entropy_matrix[t, :len(m["entropy"])] = m["entropy"]
            gini_matrix[t, :len(m["gini"])] = m["gini"]
            defector_matrix[t, :len(m["defector_ratio"])] = m["defector_ratio"]
            if m["time_to_convergence"] is not None:
                convergence_times.append(m["time_to_convergence"])
            all_change_points.extend(m["change_points"])
        
        return {
            "mean_cooperation": np.mean(coop_matrix, axis=0).tolist(),
            "std_cooperation": np.std(coop_matrix, axis=0).tolist(),
            "mean_entropy": np.mean(entropy_matrix, axis=0).tolist(),
            "mean_gini": np.mean(gini_matrix, axis=0).tolist(),
            "mean_defector_ratio": np.mean(defector_matrix, axis=0).tolist(),
            "mean_time_to_convergence": float(np.mean(convergence_times)) if convergence_times else None,
            "convergence_rate": len(convergence_times) / n_trials,
            "total_change_points": len(all_change_points),
        }
    
    # ================================
    # Signaling Game helpers
    # ================================
    
    def _run_signaling_trial(self, config, agent_type):
        """Run a single trial of the Signaling Game."""
        from environments.signaling_game import SignalingGame
        
        num_senders = config["num_senders"]
        num_receivers = config["num_receivers"]
        topology = config["topology"]
        high_prob = config.get("high_probability", 0.5)
        mem_limit = config["memory_limit"]
        mixed_mode = agent_type == "llm_sender_adaptive_receiver"
        
        # Create agents
        agents = []
        for i in range(num_senders):
            if agent_type == "llm" or mixed_mode:
                agent = SenderAgent(id=f"Sender_{i}", client=self.client, memory_limit=mem_limit)
            else:
                sender_class = SENDER_BASELINE_TYPES.get(agent_type)
                if sender_class is None:
                    raise ValueError(f"Unknown sender agent_type: {agent_type}")
                agent = sender_class(id=f"Sender_{i}", memory_limit=mem_limit)
            agents.append(agent)
        
        for i in range(num_receivers):
            if agent_type == "llm":
                agent = ReceiverAgent(id=f"Receiver_{i}", client=self.client, memory_limit=mem_limit)
            elif mixed_mode:
                receiver_class = RECEIVER_BASELINE_TYPES["adaptive_trust"]
                agent = receiver_class(id=f"Receiver_{i}", memory_limit=mem_limit)
            else:
                receiver_class = RECEIVER_BASELINE_TYPES.get(agent_type)
                if receiver_class is None:
                    raise ValueError(f"Unknown receiver agent_type: {agent_type}")
                agent = receiver_class(id=f"Receiver_{i}", memory_limit=mem_limit)
            agents.append(agent)
        
        # Create environment
        env = SignalingGame(
            num_senders=num_senders,
            num_receivers=num_receivers,
            topology=topology,
            high_prob=high_prob,
            receiver_action_mode=config.get("receiver_action_mode", "global"),
            reveal_sender_states=config.get("reveal_sender_states", True),
            sender_reward_high_invest=config.get("sender_reward_high_invest", config.get("sender_reward_invest", 3.0)),
            sender_reward_low_invest=config.get("sender_reward_low_invest", config.get("sender_reward_invest", 3.0)),
            sender_reward_pass=config.get("sender_reward_pass", 0.0),
            receiver_reward_high_invest=config.get("receiver_reward_high_invest", 3.0),
            receiver_reward_low_invest=config.get("receiver_reward_low_invest", -1.0),
            receiver_reward_pass=config.get("receiver_reward_pass", 0.0),
        )
        
        runner = SimulationRunner(agents, env, steps=config["rounds"])
        return runner.run()
    
    def _compute_signaling_metrics(self, history):
        """Compute signaling game metrics for a single trial."""
        deception = metrics.compute_deception_rate(history)
        trust = metrics.compute_trust_rate(history)
        informed_trust = metrics.compute_informed_trust_rate(history)
        return {
            "deception_rate": deception,
            "trust_rate": trust,
            "informed_trust_rate": informed_trust,
            "deception_success": metrics.compute_deception_success_rate(history),
            "receiver_accuracy": metrics.compute_receiver_accuracy(history),
            "stability_index": metrics.compute_stability_index(deception),
            "time_to_convergence": metrics.compute_time_to_convergence(deception),
            "change_points": metrics.detect_change_point(deception),
        }
    
    def _aggregate_signaling_metrics(self, trials_data, config):
        """Aggregate signaling game metrics across trials."""
        rounds = config["rounds"]
        n_trials = config["trials"]
        
        deception_matrix = np.zeros((n_trials, rounds))
        trust_matrix = np.zeros((n_trials, rounds))
        success_matrix = np.zeros((n_trials, rounds))
        accuracy_matrix = np.zeros((n_trials, rounds))
        convergence_times = []
        all_change_points = []
        
        # informed_trust can have None values (rounds with no informed receivers)
        informed_trust_lists = []
        
        for t, data in enumerate(trials_data):
            m = data["metrics"]
            deception_matrix[t, :len(m["deception_rate"])] = m["deception_rate"]
            trust_matrix[t, :len(m["trust_rate"])] = m["trust_rate"]
            success_matrix[t, :len(m["deception_success"])] = m["deception_success"]
            accuracy_matrix[t, :len(m["receiver_accuracy"])] = m["receiver_accuracy"]
            informed_trust_lists.append(m.get("informed_trust_rate", []))
            if m["time_to_convergence"] is not None:
                convergence_times.append(m["time_to_convergence"])
            all_change_points.extend(m["change_points"])
        
        # Aggregate informed_trust_rate: average across trials, skipping None values per round
        mean_informed_trust = []
        for r in range(rounds):
            vals = [t_list[r] for t_list in informed_trust_lists
                    if r < len(t_list) and t_list[r] is not None]
            mean_informed_trust.append(float(np.mean(vals)) if vals else None)
        
        return {
            "mean_deception_rate": np.mean(deception_matrix, axis=0).tolist(),
            "std_deception_rate": np.std(deception_matrix, axis=0).tolist(),
            "mean_trust_rate": np.mean(trust_matrix, axis=0).tolist(),
            "std_trust_rate": np.std(trust_matrix, axis=0).tolist(),
            "mean_informed_trust_rate": mean_informed_trust,
            "mean_deception_success": np.mean(success_matrix, axis=0).tolist(),
            "mean_receiver_accuracy": np.mean(accuracy_matrix, axis=0).tolist(),
            "mean_time_to_convergence": float(np.mean(convergence_times)) if convergence_times else None,
            "convergence_rate": len(convergence_times) / n_trials,
            "total_change_points": len(all_change_points),
        }
