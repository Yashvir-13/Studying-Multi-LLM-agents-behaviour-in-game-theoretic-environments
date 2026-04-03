import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class SimulationRunner:
    """
    Manages the execution of a single simulation run.
    
    Responsibilities:
    - Agent-environment interaction loop
    - Collecting and storing interaction history
    - Handling perturbations (sudden environment changes)
    - Contribution validation and clamping
    - Configurable observation types for information control
    """
    def __init__(self, agents, environment, steps, observation_type="full"):
        """
        Args:
            agents (list): List of instantiated Agent objects.
            environment (object): The environment instance (e.g., PublicGoodsGame).
            steps (int): Total number of rounds to simulate.
            observation_type (str): Controls what agents see about others:
                - 'full': agents see all individual contributions (default)
                - 'partial': agents only see total pool amount
                - 'private': agents only see their own reward
        """
        self.agents = agents
        self.environment = environment
        self.steps = steps
        self.history = []
        self.observation_type = observation_type
        self.validation_log = []  # Track out-of-range contributions
    def run(self, perturbation=None):
        """
        perturbation = {
            "round": 10,
            "type": "lower_multiplier" | "add_defector",
            "value": 1.2 (for multiplier) or agent_obj (for add_defector)
        }
        """
        if self.environment.name=='public_good':
            initial_endowment=self.environment.initial_endowment
            multiplier=self.environment.multiplier
            
            round_pbar = tqdm(range(self.steps), desc="    Rounds", unit="rd", leave=False)
            for i in round_pbar:
                # PERTURBATION CHECK
                if perturbation and i + 1 == perturbation["round"]:
                    print(f"\n[PERTURBATION TRIGGERED] Type: {perturbation['type']}")
                    if perturbation["type"] == "lower_multiplier":
                        multiplier = perturbation["value"]
                        self.environment.multiplier = multiplier # Update env if needed
                    elif perturbation["type"] == "reduce_memory":
                        new_limit = perturbation["value"]
                        for ag in self.agents:
                            if hasattr(ag, 'memory_limit'):
                                ag.memory_limit = new_limit
                
                actions = {}
                # Previous round data for observation
                last_round = self.history[-1] if self.history else None
                
                # Build observations based on observation_type
                observations = {}
                for agent in self.agents:
                    obs = {
                        "round_num": i + 1,
                        "config": {
                            "initial_endowment": initial_endowment,
                            "multiplier": multiplier,
                            "num_agents": len(self.agents)
                        },
                        "my_cumulative_reward": sum(h['rewards'].get(agent.id, 0) for h in self.history) if self.history else 0
                    }
                    
                    # Control information structure
                    if self.observation_type == "full" and last_round:
                        obs["prev_round_actions"] = last_round['actions']
                        obs["prev_round_rewards"] = last_round['rewards']
                    elif self.observation_type == "partial" and last_round:
                        # Only see total pool and own reward
                        obs["prev_round_total_contribution"] = sum(last_round['actions'].values())
                        obs["prev_round_my_reward"] = last_round['rewards'].get(agent.id, 0)
                    elif self.observation_type == "private" and last_round:
                        # Only see own reward
                        obs["prev_round_my_reward"] = last_round['rewards'].get(agent.id, 0)
                    
                    observations[agent.id] = obs
                
                # Call all agents in PARALLEL
                def _call_agent(agent):
                    obs = observations[agent.id]
                    action_response = agent.act(obs)
                    if isinstance(action_response, dict) and "contribution" in action_response:
                        raw_val = int(action_response["contribution"])
                        # CLAMP to valid range [0, endowment]
                        clamped = max(0, min(initial_endowment, raw_val))
                        if raw_val != clamped:
                            self.validation_log.append({
                                "round": i + 1,
                                "agent": agent.id,
                                "raw": raw_val,
                                "clamped": clamped
                            })
                        return agent.id, clamped
                    return agent.id, 0
                
                with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                    futures = [executor.submit(_call_agent, agent) for agent in self.agents]
                    for future in as_completed(futures):
                        agent_id, contribution = future.result()
                        actions[agent_id] = contribution

                rewards=self.environment.step(actions)
                
                # Update progress bar with summary stats
                avg_contrib = sum(actions.values()) / len(actions) if actions else 0
                round_pbar.set_postfix_str(f"avg={avg_contrib:.1f}, pool={self.environment.last_round_pool}")
                
                round_record = {
                    "round": i+1,
                    "actions": actions,
                    "rewards": rewards,
                    "round_pool": self.environment.last_round_pool,
                    "cumulative_pool": self.environment.cumulative_pool
                }
                for agent in self.agents:
                    agent.update(round_record)
                self.history.append(round_record)
            return self.history
        
        elif self.environment.name == 'signaling_game':
            # ============================
            # SIGNALING GAME LOOP
            # ============================
            # Agents are split into senders and receivers
            senders = [a for a in self.agents if a.role == "sender"]
            receivers = [a for a in self.agents if a.role == "receiver"]
            topology_links = self.environment.get_topology_links()
            action_mode = getattr(self.environment, "receiver_action_mode", "global")
            reveal_sender_states = getattr(self.environment, "reveal_sender_states", True)
            payoff_config = {
                f"{state}_{action}": {
                    "sender": vals[0],
                    "receiver": vals[1],
                }
                for (state, action), vals in getattr(self.environment, "payoffs", {}).items()
            }
            
            round_pbar = tqdm(range(self.steps), desc="    Rounds", unit="rd", leave=False)
            for i in round_pbar:
                
                # Phase 1: Nature assigns private states to senders
                states = self.environment.assign_states()
                
                # Phase 2: Build sender observations and collect messages (parallel)
                sender_observations = {}
                for sender in senders:
                    obs = {
                        "round_num": i + 1,
                        "private_state": states[sender.id],
                        "connected_receivers": [rid for rid, sids in topology_links.items() if sender.id in sids],
                        "config": {
                            "num_senders": self.environment.num_senders,
                            "num_receivers": self.environment.num_receivers,
                            "topology": self.environment.topology,
                            "high_probability": self.environment.high_prob,
                            "payoffs": payoff_config,
                        },
                        "my_cumulative_reward": self.environment.cumulative_sender_rewards.get(sender.id, 0),
                    }
                    sender_observations[sender.id] = obs
                
                messages = {}
                
                def _call_sender(sender):
                    obs = sender_observations[sender.id]
                    result = sender.act(obs)
                    msg = result.get("message", states[sender.id])
                    # Validate
                    if msg not in ("high", "low"):
                        msg = states[sender.id]  # default to truth
                    return sender.id, msg
                
                with ThreadPoolExecutor(max_workers=max(1, len(senders))) as executor:
                    futures = [executor.submit(_call_sender, s) for s in senders]
                    for future in as_completed(futures):
                        sid, msg = future.result()
                        messages[sid] = msg
                
                # Phase 3: Route messages to receivers via topology
                receiver_observations = {}
                for receiver in receivers:
                    connected_senders = topology_links.get(receiver.id, [])
                    inbox = {sid: messages[sid] for sid in connected_senders if sid in messages}
                    
                    obs = {
                        "round_num": i + 1,
                        "messages": inbox,
                        "action_mode": action_mode,
                        "config": {
                            "num_senders": self.environment.num_senders,
                            "num_receivers": self.environment.num_receivers,
                            "topology": self.environment.topology,
                            "receiver_action_mode": action_mode,
                            "high_probability": self.environment.high_prob,
                            "payoffs": payoff_config,
                        },
                        "my_cumulative_reward": self.environment.cumulative_receiver_rewards.get(receiver.id, 0),
                    }
                    receiver_observations[receiver.id] = obs
                
                # Phase 4: Receivers decide (parallel)
                actions = {}
                
                def _call_receiver(receiver):
                    obs = receiver_observations[receiver.id]
                    result = receiver.act(obs)
                    if action_mode == "per_sender":
                        connected = topology_links.get(receiver.id, [])
                        action_map = result.get("actions", {})
                        if not isinstance(action_map, dict):
                            action_map = {}
                        validated = {}
                        for sid in connected:
                            act = action_map.get(sid, "pass")
                            validated[sid] = act if act in ("invest", "pass") else "pass"
                        return receiver.id, validated

                    act = result.get("action", "pass")
                    if act not in ("invest", "pass"):
                        act = "pass"
                    return receiver.id, act
                
                with ThreadPoolExecutor(max_workers=max(1, len(receivers))) as executor:
                    futures = [executor.submit(_call_receiver, r) for r in receivers]
                    for future in as_completed(futures):
                        rid, act = future.result()
                        actions[rid] = act
                
                # Phase 5: Environment computes payoffs
                rewards = self.environment.step(states, messages, actions)
                
                # Build round record with ground truth
                round_record = self.environment.get_round_summary(states, messages, actions, rewards)
                
                # Progress bar
                n_lied = sum(1 for sid in senders if messages.get(sid.id) != states.get(sid.id))
                if action_mode == "per_sender":
                    n_invested = sum(
                        1 for receiver_actions in actions.values()
                        for act in receiver_actions.values()
                        if act == "invest"
                    )
                else:
                    n_invested = sum(1 for act in actions.values() if act == "invest")
                round_pbar.set_postfix_str(f"lies={n_lied}, invest={n_invested}")
                
                # Update all agents with round record
                # Build role-specific views for memory
                for sender in senders:
                    receiver_actions_for_me = {}
                    for rid, receiver_actions in actions.items():
                        if isinstance(receiver_actions, dict):
                            receiver_actions_for_me[rid] = receiver_actions.get(sender.id, "pass")
                        else:
                            receiver_actions_for_me[rid] = receiver_actions
                    sender_view = {
                        "round": i + 1,
                        "my_state": states[sender.id],
                        "my_message": messages[sender.id],
                        "my_reward": rewards.get(sender.id, 0),
                        "receiver_actions_for_me": receiver_actions_for_me,
                    }
                    sender.update(sender_view)
                
                for receiver in receivers:
                    connected = topology_links.get(receiver.id, [])
                    receiver_actions = actions.get(receiver.id, {} if action_mode == "per_sender" else "pass")
                    receiver_view = {
                        "round": i + 1,
                        "messages_received": {sid: messages[sid] for sid in connected if sid in messages},
                        "my_actions": receiver_actions,
                        "my_reward": rewards.get(receiver.id, 0),
                    }
                    if reveal_sender_states:
                        receiver_view["true_states"] = {sid: states[sid] for sid in connected if sid in states}
                        receiver_view["sender_honesty"] = {
                            sid: (messages.get(sid) == states.get(sid))
                            for sid in connected if sid in messages and sid in states
                        }
                    receiver.update(receiver_view)
                
                self.history.append(round_record)
            
            return self.history
