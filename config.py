"""
Central Experiment Configuration.

All experiment parameters live here — no more scattered hardcoded values.
Load from YAML or JSON, pass everywhere, auto-save to log directories.
"""

import json
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional
from copy import deepcopy


@dataclass
class ExperimentConfig:
    """
    Central configuration for all experiments.
    
    Usage:
        # From YAML
        config = ExperimentConfig.from_yaml("my_experiment.yaml")
        
        # From code (testing)
        config = ExperimentConfig(rounds=10, trials_per_config=5)
        
        # Save to output directory
        config.save("logs/sweep_20260223/")
    """
    
    # === Model Settings ===
    model: str = "qwen3.5:9b"
    temperature: float = 0.7
    num_predict: int = 64
    think: Optional[Any] = False
    
    # === Environment Selector ===
    environment: str = "public_good"     # "public_good" | "signaling_game"
    
    # === Public Goods Sweep Grid (lists → cartesian product) ===
    num_agents: List[int] = field(default_factory=lambda: [3, 5])
    multiplier: List[float] = field(default_factory=lambda: [1.2, 2.0])
    memory_limit: List[int] = field(default_factory=lambda: [1, 3])
    
    # === Signaling Game Parameters ===
    num_senders: List[int] = field(default_factory=lambda: [2])
    num_receivers: List[int] = field(default_factory=lambda: [2])
    topology: List[str] = field(default_factory=lambda: ["full"])
    high_probability: float = 0.5
    receiver_action_mode: str = "per_sender"   # "global" | "per_sender"
    reveal_sender_states: bool = True
    sender_reward_high_invest: float = 3.0
    sender_reward_low_invest: float = 3.0
    sender_reward_invest: float = 3.0
    sender_reward_pass: float = 0.0
    receiver_reward_high_invest: float = 3.0
    receiver_reward_low_invest: float = -1.0
    receiver_reward_pass: float = 0.0
    
    # === Simulation Parameters ===
    rounds: int = 20
    trials_per_config: int = 15
    initial_endowment: int = 10
    observation_type: str = "full"       # "full" | "partial" | "private"
    agent_type: str = "llm"              # "llm" | baseline types
    
    # === Output ===
    base_output_dir: str = "logs"
    
    # === Resume ===
    resume_dir: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        """Auto-detect format and load."""
        if path.endswith((".yaml", ".yml")):
            return cls.from_yaml(path)
        elif path.endswith(".json"):
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path}. Use .yaml or .json")
    
    def save(self, directory: str, filename: str = "experiment_config.yaml"):
        """Save this config to a directory for reproducibility."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        return path
    
    def to_dict(self) -> dict:
        """Convert to plain dict (for JSON serialization)."""
        return asdict(self)
    
    def get_sweep_config(self) -> dict:
        """
        Returns the sweep config dict expected by ExperimentRunner.
        Works for both public_good and signaling_game environments.
        """
        base = {
            "environment": self.environment,
            "memory_limit": self.memory_limit,
            "rounds": self.rounds,
            "trials_per_config": self.trials_per_config,
            "agent_type": self.agent_type,
        }
        
        if self.environment == "signaling_game":
            base.update({
                "num_senders": self.num_senders,
                "num_receivers": self.num_receivers,
                "topology": self.topology,
                "high_probability": self.high_probability,
                "receiver_action_mode": self.receiver_action_mode,
                "reveal_sender_states": self.reveal_sender_states,
                "sender_reward_high_invest": self.sender_reward_high_invest,
                "sender_reward_low_invest": self.sender_reward_low_invest,
                "sender_reward_invest": self.sender_reward_invest,
                "sender_reward_pass": self.sender_reward_pass,
                "receiver_reward_high_invest": self.receiver_reward_high_invest,
                "receiver_reward_low_invest": self.receiver_reward_low_invest,
                "receiver_reward_pass": self.receiver_reward_pass,
            })
        else:
            base.update({
                "num_agents": self.num_agents,
                "multiplier": self.multiplier,
                "initial_endowment": self.initial_endowment,
                "observation_type": self.observation_type,
            })
        
        return base
    def __repr__(self):
        lines = ["ExperimentConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
