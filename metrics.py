import numpy as np
from collections import Counter

def compute_cooperation_rate(history, endowment):
    """
    Calculates the average cooperation rate per round.
    Cooperation Rate = (Total Contribution) / (Num Agents * Endowment)
    Returns: List of floats [0.0 - 1.0] for each round.
    """
    rates = []
    for round_data in history:
        actions = round_data.get("actions", {})
        if not actions:
            rates.append(0.0)
            continue
        
        total_contribution = sum(actions.values())
        max_possible = len(actions) * endowment
        rates.append(total_contribution / max_possible if max_possible > 0 else 0.0)
    return rates

def compute_defector_ratio(history):
    """
    Calculates the fraction of agents contributing 0 per round.
    """
    ratios = []
    for round_data in history:
        actions = round_data.get("actions", {})
        if not actions:
            ratios.append(0.0)
            continue
            
        defector_count = sum(1 for val in actions.values() if val == 0)
        ratios.append(defector_count / len(actions))
    return ratios

def compute_behavior_entropy(history, bins=11):
    """
    Calculates Shannon entropy of contribution distribution per round.
    Assuming integer contributions 0-10 (11 bins).
    """
    entropies = []
    for round_data in history:
        actions = list(round_data.get("actions", {}).values())
        if not actions:
            entropies.append(0.0)
            continue
            
        counts = Counter(actions)
        total = len(actions)
        probs = [count / total for count in counts.values()]
        
        # Shannon Entropy H = -sum(p * log2(p))
        ent = -sum(p * np.log2(p) for p in probs if p > 0)
        entropies.append(ent)
    return entropies

def compute_reward_gini(history):
    """
    Calculates Gini coefficient of rewards per round.
    0 = Perfect Equality, 1 = Perfect Inequality.
    """
    ginis = []
    for round_data in history:
        rewards = list(round_data.get("rewards", {}).values())
        if not rewards or sum(rewards) == 0:
            ginis.append(0.0)
            continue
            
        # Gini calculation
        sorted_rewards = sorted(rewards)
        n = len(rewards)
        cumulative = np.cumsum(sorted_rewards)
        sum_rewards = cumulative[-1]
        
        # Formula: G = (n+1)/n - 2 * sum(cumulative) / (n * sum_rewards)
        gini = (n + 1) / n - 2 * sum(cumulative) / (n * sum_rewards)
        ginis.append(max(0.0, gini)) # Clamp to 0
    return ginis

def compute_stability_index(cooperation_rates, window=3):
    """
    Calculates Variance of cooperation rate over a rolling window.
    Lower variance = Higher stability.
    Returns: List of variances (len = len(rates) - window + 1)
    """
    if len(cooperation_rates) < window:
        return [0.0]
        
    variances = []
    for i in range(len(cooperation_rates) - window + 1):
        window_slice = cooperation_rates[i : i + window]
        variances.append(np.var(window_slice))
    return variances

def detect_change_point(cooperation_rates, threshold=0.2):
    """
    Simple change point detection: looks for sudden shifts > threshold between rounds.
    Returns: List of round indices where shift occurred.
    """
    change_points = []
    for i in range(1, len(cooperation_rates)):
        delta = abs(cooperation_rates[i] - cooperation_rates[i-1])
        if delta > threshold:
            change_points.append(i + 1) # 1-based round index
    return change_points

def compute_time_to_convergence(cooperation_rates, threshold=0.02, window=3):
    """
    Measures the round at which the cooperation rate first stabilizes.
    
    Convergence = the first round where the variance of the cooperation rate
    over the next 'window' rounds is below 'threshold'.
    
    Args:
        cooperation_rates (list): Per-round cooperation rates.
        threshold (float): Maximum variance to consider "converged" (default: 0.02).
        window (int): Number of consecutive rounds to check stability (default: 3).
    
    Returns:
        int or None: Round number (1-based) at which convergence occurs, or None if never.
    """
    if len(cooperation_rates) < window:
        return None
        
    for i in range(len(cooperation_rates) - window + 1):
        window_slice = cooperation_rates[i : i + window]
        if np.var(window_slice) < threshold:
            return i + 1  # 1-based round
    return None


# ===========================
# Signaling Game Metrics
# ===========================

def compute_deception_rate(history):
    """
    Fraction of sender messages that differ from the true state per round.
    Returns: List of floats [0.0 - 1.0] for each round.
    """
    rates = []
    for round_data in history:
        deception_log = round_data.get("deception_log", {})
        if not deception_log:
            rates.append(0.0)
            continue
        lied_count = sum(1 for v in deception_log.values() if v.get("lied", False))
        rates.append(lied_count / len(deception_log))
    return rates


def compute_trust_rate(history):
    """
    Fraction of ALL receivers (including empty-inbox) that chose "invest" per round.
    This is the aggregate trust rate — includes receivers who had no information.
    Returns: List of floats [0.0 - 1.0] for each round.
    """
    rates = []
    for round_data in history:
        actions = round_data.get("actions", {})
        topology_links = round_data.get("topology_links", {})
        receiver_actions = {k: v for k, v in actions.items() if k.startswith("Receiver")}
        if not receiver_actions:
            rates.append(0.0)
            continue

        if any(isinstance(v, dict) for v in receiver_actions.values()):
            total_edges = 0
            invested = 0
            for rid, action_map in receiver_actions.items():
                connected = topology_links.get(rid, [])
                if not isinstance(action_map, dict):
                    continue
                for sid in connected:
                    total_edges += 1
                    if action_map.get(sid) == "invest":
                        invested += 1
            rates.append(invested / total_edges if total_edges else 0.0)
            continue

        invested = sum(1 for v in receiver_actions.values() if v == "invest")
        rates.append(invested / len(receiver_actions))
    return rates


def compute_informed_trust_rate(history):
    """
    Fraction of INFORMED receivers (those who received >=1 message) that chose "invest".
    Excludes receivers with empty inboxes — they have no basis for a trust decision.
    In star topology, this gives the true trust rate of the central receiver only.
    Returns: List of floats [0.0 - 1.0] for each round (None if no informed receivers).
    """
    rates = []
    for round_data in history:
        actions = round_data.get("actions", {})
        topology_links = round_data.get("topology_links", {})

        informed_receivers = {
            rid: act for rid, act in actions.items()
            if rid.startswith("Receiver") and len(topology_links.get(rid, [])) > 0
        }

        if not informed_receivers:
            rates.append(None)  # No informed receivers this round
            continue

        if any(isinstance(v, dict) for v in informed_receivers.values()):
            total_edges = 0
            invested = 0
            for rid, action_map in informed_receivers.items():
                connected = topology_links.get(rid, [])
                if not isinstance(action_map, dict):
                    continue
                for sid in connected:
                    total_edges += 1
                    if action_map.get(sid) == "invest":
                        invested += 1
            rates.append(invested / total_edges if total_edges else None)
            continue

        invested = sum(1 for v in informed_receivers.values() if v == "invest")
        rates.append(invested / len(informed_receivers))
    return rates


def compute_deception_success_rate(history):
    """
    Topology-aware: of rounds where a sender lied, what fraction of receivers
    CONNECTED TO THAT LIAR invested?
    
    Only counts receivers who actually received a deceptive message.
    Returns: List of floats per round (0.0 if no deception that round).
    """
    rates = []
    for round_data in history:
        deception_log = round_data.get("deception_log", {})
        actions = round_data.get("actions", {})
        topology_links = round_data.get("topology_links", {})

        liars = {sid for sid, info in deception_log.items() if info.get("lied", False)}
        if not liars:
            rates.append(0.0)
            continue

        # Build reverse map: sender → list of receivers it sends to
        sender_to_receivers = {}
        for rid, senders in topology_links.items():
            for sid in senders:
                sender_to_receivers.setdefault(sid, []).append(rid)

        # Find all receivers connected to at least one liar
        duped_receivers = set()
        for liar in liars:
            duped_receivers.update(sender_to_receivers.get(liar, []))

        if not duped_receivers:
            rates.append(0.0)
            continue

        edge_level_actions = any(isinstance(v, dict) for v in actions.values())
        if edge_level_actions:
            total_targets = 0
            invested = 0
            for liar in liars:
                for rid in sender_to_receivers.get(liar, []):
                    total_targets += 1
                    receiver_actions = actions.get(rid, {})
                    if isinstance(receiver_actions, dict) and receiver_actions.get(liar) == "invest":
                        invested += 1
            rates.append(invested / total_targets if total_targets else 0.0)
        else:
            invested = sum(1 for rid in duped_receivers if actions.get(rid) == "invest")
            rates.append(invested / len(duped_receivers))
    return rates


def compute_receiver_accuracy(history):
    """
    Topology-aware: fraction of INFORMED receivers whose action was optimal per round.
    
    Optimal for receiver R = invest if ALL senders CONNECTED TO R have true state "high",
    pass if any connected sender has true state "low".
    Receivers with no connected senders are excluded (no information to act on).
    
    Returns: List of floats [0.0 - 1.0] for each round.
    """
    rates = []
    for round_data in history:
        states = round_data.get("states", {})
        actions = round_data.get("actions", {})
        topology_links = round_data.get("topology_links", {})

        informed_receivers = {
            rid: act for rid, act in actions.items()
            if rid.startswith("Receiver") and len(topology_links.get(rid, [])) > 0
        }

        if not informed_receivers:
            rates.append(0.0)
            continue

        if any(isinstance(v, dict) for v in informed_receivers.values()):
            total_edges = 0
            correct = 0
            for rid, action_map in informed_receivers.items():
                connected_senders = topology_links.get(rid, [])
                if not isinstance(action_map, dict):
                    continue
                for sid in connected_senders:
                    total_edges += 1
                    optimal = "invest" if states.get(sid, "low") == "high" else "pass"
                    if action_map.get(sid, "pass") == optimal:
                        correct += 1
            rates.append(correct / total_edges if total_edges else 0.0)
        else:
            correct = 0
            for rid, action in informed_receivers.items():
                connected_senders = topology_links.get(rid, [])
                # Optimal: invest only if ALL connected senders have high true state
                connected_states = [states.get(sid, "low") for sid in connected_senders]
                optimal = "invest" if all(s == "high" for s in connected_states) else "pass"
                if action == optimal:
                    correct += 1

            rates.append(correct / len(informed_receivers))
    return rates


def compute_truth_telling_distance(history):
    """
    Distance from perfect truth-telling equilibrium per round.
    0.0 = all senders told truth, 1.0 = all senders lied.
    Same as deception_rate but named for game-theoretic clarity.
    """
    return compute_deception_rate(history)
