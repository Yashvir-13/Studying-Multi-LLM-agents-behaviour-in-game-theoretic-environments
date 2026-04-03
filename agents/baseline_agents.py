
import random

class BaselineAgent:
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
    
    def update(self, result):
        self.memory.append(result)
        
    def act(self, observation):
        """To be overridden"""
        return {"contribution": 0}

class AlwaysCooperateAgent(BaselineAgent):
    def act(self, observation):
        # Always contribute max (assuming 10 for now, or get from config)
        endowment = observation.get("config", {}).get("initial_endowment", 10)
        return {"contribution": endowment}

class AlwaysDefectAgent(BaselineAgent):
    def act(self, observation):
        return {"contribution": 0}

class RandomAgent(BaselineAgent):
    def act(self, observation):
        endowment = observation.get("config", {}).get("initial_endowment", 10)
        return {"contribution": random.randint(0, endowment)}

class TitForTatAgent(BaselineAgent):
    def act(self, observation):
        # First move: Cooperate
        if not self.memory:
            endowment = observation.get("config", {}).get("initial_endowment", 10)
            return {"contribution": endowment}
            
        # Subsequent moves: Copy the average contribution of others in the last round?
        # Or standard TFT: if opponent cooperated, I cooperate.
        # In N-player public goods, TFT is harder to define.
        # Common adaptation: Cooperate if group cooperation > threshold, else Defect.
        # OR: Copy the group's average contribution.
        
        prev_actions = observation.get("prev_round_actions", {})
        if not prev_actions:
             endowment = observation.get("config", {}).get("initial_endowment", 10)
             return {"contribution": endowment}
        
        # Filter out self? or include self? usually others.
        others_contributions = [val for ag_id, val in prev_actions.items() if ag_id != self.id]
        
        if not others_contributions:
             return {"contribution": 0}
             
        avg_others = sum(others_contributions) / len(others_contributions)
        return {"contribution": int(avg_others)}
