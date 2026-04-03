"""
Baseline (non-LLM) agents for the Signaling Game.

Used for control experiments and comparison against LLM behavior.
"""

import random


# ========================
# Sender Baselines
# ========================

class AlwaysTruthSender:
    """Always sends the true state as the message."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "sender"
    
    def act(self, observation):
        return {"message": observation.get("private_state", "low")}
    
    def update(self, result):
        self.memory.append(result)


class AlwaysLieSender:
    """Always sends the opposite of the true state."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "sender"
    
    def act(self, observation):
        state = observation.get("private_state", "low")
        return {"message": "low" if state == "high" else "high"}
    
    def update(self, result):
        self.memory.append(result)


class RandomSender:
    """Sends a random message regardless of true state."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "sender"
    
    def act(self, observation):
        return {"message": random.choice(["high", "low"])}
    
    def update(self, result):
        self.memory.append(result)


class StrategicLiarSender:
    """Tells truth when state is high, lies when state is low (optimal deception)."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "sender"
    
    def act(self, observation):
        # Always claim high — truth when high, lie when low
        return {"message": "high"}
    
    def update(self, result):
        self.memory.append(result)


# ========================
# Receiver Baselines
# ========================

class AlwaysInvestReceiver:
    """Always invests regardless of messages."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
    
    def act(self, observation):
        return {"action": "invest"}
    
    def update(self, result):
        self.memory.append(result)


class AlwaysPassReceiver:
    """Always passes regardless of messages."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
    
    def act(self, observation):
        return {"action": "pass"}
    
    def update(self, result):
        self.memory.append(result)


class TrustingReceiver:
    """Invests if any sender reports 'high', passes otherwise."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
    
    def act(self, observation):
        messages = observation.get("messages", {})
        if any(m == "high" for m in messages.values()):
            return {"action": "invest"}
        return {"action": "pass"}
    
    def update(self, result):
        self.memory.append(result)


class SkepticalReceiver:
    """Invests only if ALL senders report 'high'."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
    
    def act(self, observation):
        messages = observation.get("messages", {})
        if messages and all(m == "high" for m in messages.values()):
            return {"action": "invest"}
        return {"action": "pass"}
    
    def update(self, result):
        self.memory.append(result)


class RandomReceiver:
    """Randomly invests or passes."""
    def __init__(self, id, memory_limit=5):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
    
    def act(self, observation):
        return {"action": random.choice(["invest", "pass"])}
    
    def update(self, result):
        self.memory.append(result)


class AdaptiveTrustReceiver:
    """
    Reputation-based receiver for quick signaling-game demos.

    Starts moderately trusting, then updates sender-specific trust from revealed
    honesty feedback in memory. Invests on HIGH messages from sufficiently trusted
    senders and passes otherwise.
    """
    def __init__(self, id, memory_limit=5, prior_truth=0.75, trust_threshold=0.55):
        self.id = id
        self.memory = []
        self.memory_limit = memory_limit
        self.role = "receiver"
        self.prior_truth = prior_truth
        self.trust_threshold = trust_threshold

    def _estimate_trust(self, sender_id):
        recent_memory = self.memory[-self.memory_limit:] if self.memory_limit > 0 else self.memory
        truthful = 0
        total = 0
        for item in recent_memory:
            sender_honesty = item.get("sender_honesty", {})
            if sender_id in sender_honesty:
                truthful += 1 if sender_honesty[sender_id] else 0
                total += 1

        # Beta-style smoothing keeps early rounds from collapsing to pure caution.
        pseudo_total = 2
        pseudo_truthful = self.prior_truth * pseudo_total
        return (truthful + pseudo_truthful) / (total + pseudo_total)

    def act(self, observation):
        messages = observation.get("messages", {})
        action_mode = observation.get("action_mode", "global")

        if action_mode == "per_sender":
            actions = {}
            for sender_id, message in messages.items():
                trust = self._estimate_trust(sender_id)
                actions[sender_id] = "invest" if message == "high" and trust >= self.trust_threshold else "pass"
            return {"actions": actions}

        for sender_id, message in messages.items():
            trust = self._estimate_trust(sender_id)
            if message == "high" and trust >= self.trust_threshold:
                return {"action": "invest"}
        return {"action": "pass"}

    def update(self, result):
        self.memory.append(result)


# Maps for experiment runner
SENDER_BASELINE_TYPES = {
    "always_truth": AlwaysTruthSender,
    "always_lie": AlwaysLieSender,
    "random_sender": RandomSender,
    "strategic_liar": StrategicLiarSender,
}

RECEIVER_BASELINE_TYPES = {
    "always_invest": AlwaysInvestReceiver,
    "always_pass": AlwaysPassReceiver,
    "trusting": TrustingReceiver,
    "skeptical": SkepticalReceiver,
    "random_receiver": RandomReceiver,
    "adaptive_trust": AdaptiveTrustReceiver,
}
