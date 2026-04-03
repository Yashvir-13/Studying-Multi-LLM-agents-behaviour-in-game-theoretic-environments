"""
Signaling Game (Information Asymmetry Environment)

Nature assigns hidden states to Senders. Senders transmit messages to Receivers
via a configurable communication topology. Receivers make investment decisions
based on the signals they receive. Payoffs depend on the TRUE state, not the message.

Emergent behaviors: truth-telling equilibria, strategic deception, coalition-based misinformation.
"""

import random
from collections import defaultdict


class SignalingGame:
    """
    Environment for a repeated signaling game with private information.
    
    Each round:
    1. Nature assigns a state ("high" or "low") to each sender
    2. Senders observe their own state and send a message to connected receivers
    3. Receivers observe messages from connected senders and choose "invest" or "pass"
    4. Payoffs are computed based on the TRUE state and receiver's action
    
    Payoff Matrix (per sender-receiver pair):
        State=High, Action=Invest → Sender +3, Receiver +3
        State=High, Action=Pass  → Sender +0, Receiver +1
        State=Low,  Action=Invest → Sender +3, Receiver -2
        State=Low,  Action=Pass  → Sender +0, Receiver +1
    
    Communication Topologies:
        full:      Every sender → every receiver
        star:      All senders → one central receiver (requires num_receivers=1)
        ring:      Each sender → one adjacent receiver (1:1 pairing, wraps around)
        broadcast: First sender → all receivers (requires num_senders=1)
    """
    
    def __init__(self, num_senders, num_receivers, topology="full", high_prob=0.5,
                 receiver_action_mode="global", reveal_sender_states=True,
                 sender_reward_high_invest=3.0, sender_reward_low_invest=3.0, sender_reward_pass=0.0,
                 receiver_reward_high_invest=3.0, receiver_reward_low_invest=-1.0,
                 receiver_reward_pass=0.0):
        """
        Args:
            num_senders (int): Number of sender agents.
            num_receivers (int): Number of receiver agents.
            topology (str): Communication topology — "full", "star", "ring", "broadcast".
            high_prob (float): Probability that nature assigns "high" state (default: 0.5).
        """
        self.name = "signaling_game"
        self.num_senders = num_senders
        self.num_receivers = num_receivers
        self.topology = topology
        self.high_prob = high_prob
        self.receiver_action_mode = receiver_action_mode
        self.reveal_sender_states = reveal_sender_states
        self.payoffs = {
            ("high", "invest"): (sender_reward_high_invest, receiver_reward_high_invest),
            ("high", "pass"):   (sender_reward_pass, receiver_reward_pass),
            ("low",  "invest"): (sender_reward_low_invest, receiver_reward_low_invest),
            ("low",  "pass"):   (sender_reward_pass, receiver_reward_pass),
        }
        
        # IDs
        self.sender_ids = [f"Sender_{i}" for i in range(num_senders)]
        self.receiver_ids = [f"Receiver_{i}" for i in range(num_receivers)]
        
        # Validate topology constraints
        if topology == "star" and num_receivers > 1:
            raise ValueError(
                f"Star topology routes ALL senders to Receiver_0 only. "
                f"Got num_receivers={num_receivers} — extra receivers would sit idle. "
                f"Use num_receivers=1 with star, or switch to 'full' or 'ring'."
            )
        if topology == "broadcast" and num_senders > 1:
            raise ValueError(
                f"Broadcast topology only lets Sender_0 communicate. "
                f"Got num_senders={num_senders} — extra senders would be silent. "
                f"Use num_senders=1 with broadcast, or switch to 'full' or 'ring'."
            )
        
        # Build topology links once
        self._links = self._build_topology()
        
        # Tracking
        self.cumulative_sender_rewards = {}   # {sender_id: float}
        self.cumulative_receiver_rewards = {} # {receiver_id: float}
        self.round_count = 0
    
    def _build_topology(self):
        """
        Returns a dict mapping each receiver_id → list of sender_ids they hear from.
        """
        links = {rid: [] for rid in self.receiver_ids}
        
        if self.topology == "full":
            # Every sender talks to every receiver
            for rid in self.receiver_ids:
                links[rid] = list(self.sender_ids)
                
        elif self.topology == "star":
            # All senders talk to receiver_0 only
            if self.receiver_ids:
                links[self.receiver_ids[0]] = list(self.sender_ids)
                
        elif self.topology == "ring":
            # Sender_i → Receiver_(i % num_receivers)
            for i, sid in enumerate(self.sender_ids):
                rid = self.receiver_ids[i % self.num_receivers]
                links[rid].append(sid)
                
        elif self.topology == "broadcast":
            # Only Sender_0 talks to all receivers
            if self.sender_ids:
                for rid in self.receiver_ids:
                    links[rid] = [self.sender_ids[0]]
        else:
            raise ValueError(f"Unknown topology: {self.topology}. "
                             f"Use: full, star, ring, broadcast")
        
        return links
    
    def get_topology_links(self):
        """Returns receiver→[senders] mapping (read-only copy)."""
        return {k: list(v) for k, v in self._links.items()}
    
    def assign_states(self):
        """
        Nature's move: assign a hidden state to each sender.
        
        Returns:
            dict: {sender_id: "high" | "low"}
        """
        states = {}
        for sid in self.sender_ids:
            states[sid] = "high" if random.random() < self.high_prob else "low"
        return states
    
    def step(self, states, messages, actions):
        """
        Process one round of the signaling game.
        
        Args:
            states (dict): {sender_id: "high"|"low"} — ground truth from nature
            messages (dict): {sender_id: "high"|"low"} — what senders claimed
            actions (dict): receiver decisions, either:
                - global mode: {receiver_id: "invest"|"pass"}
                - per_sender mode: {receiver_id: {sender_id: "invest"|"pass"}}
            
        Returns:
            dict: {agent_id: reward} for ALL agents (senders + receivers)
        """
        self.round_count += 1
        rewards = {}
        
        # Initialize all rewards to 0
        for sid in self.sender_ids:
            rewards[sid] = 0.0
        for rid in self.receiver_ids:
            rewards[rid] = 0.0
        
        # For each receiver, compute payoffs with each connected sender
        for rid, connected_senders in self._links.items():
            receiver_actions = actions.get(rid, "pass")

            for sid in connected_senders:
                if isinstance(receiver_actions, dict):
                    action = receiver_actions.get(sid, "pass")
                else:
                    action = receiver_actions

                true_state = states.get(sid, "low")
                payoff_key = (true_state, action)
                sender_pay, receiver_pay = self.payoffs.get(payoff_key, (0, 0))
                
                rewards[sid] += sender_pay
                rewards[rid] += receiver_pay
        
        # Update cumulative tracking
        for agent_id, reward in rewards.items():
            if agent_id.startswith("Sender"):
                self.cumulative_sender_rewards[agent_id] = self.cumulative_sender_rewards.get(agent_id, 0.0) + reward
            else:
                self.cumulative_receiver_rewards[agent_id] = self.cumulative_receiver_rewards.get(agent_id, 0.0) + reward
        
        return rewards
    
    def get_round_summary(self, states, messages, actions, rewards):
        """
        Build a detailed round record for history logging.
        Includes ground truth for offline deception analysis.
        """
        # Detect deception per sender
        deception_log = {}
        for sid in self.sender_ids:
            true_state = states.get(sid)
            sent_message = messages.get(sid)
            deception_log[sid] = {
                "true_state": true_state,
                "message": sent_message,
                "lied": sent_message != true_state
            }
        
        return {
            "round": self.round_count,
            "states": states,
            "messages": messages,
            "actions": actions,
            "rewards": rewards,
            "deception_log": deception_log,
            "topology": self.topology,
            "topology_links": self.get_topology_links(),
            "receiver_action_mode": self.receiver_action_mode,
            "payoffs": {
                f"{state}_{action}": {
                    "sender": vals[0],
                    "receiver": vals[1],
                }
                for (state, action), vals in self.payoffs.items()
            },
        }
