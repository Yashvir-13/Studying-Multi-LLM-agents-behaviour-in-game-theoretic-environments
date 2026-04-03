class PublicsGood:
    """
    Simulates a standard Public Goods Game environment.
    
    Mechanics:
    1. Each agent receives an initial endowment per round.
    2. Agents decide how much to contribute to the common pool.
    3. The pool is multiplied by 'multiplier' and redistributed equally.
    
    Payoff Formula:
    Pi_i = (Endowment - Contribution_i) + (Sum(Contributions) * Multiplier) / N
    """
    def __init__(self, num_of_agents, initial_endowment, multiplier):
        """
        Args:
            num_of_agents (int): Total agents used for distribution calculation.
            initial_endowment (int): Tokens each agent starts with per round.
            multiplier (float): Factor by which the public pool is multiplied (1 < M < N).
        """
        self.num_of_agents = num_of_agents
        self.initial_endowment = initial_endowment
        self.multiplier = multiplier
        self.name = 'public_good'
        self.cumulative_pool = 0  # For logging only, NOT sent to agents
        self.last_round_pool = 0  # Current round's pool
        
    def step(self, actions):
        """
        Process one round of the game.
        
        Args:
            actions (dict): {agent_id: contribution_amount}
            
        Returns:
            dict: {agent_id: reward}
        """
        total_pool = sum(actions.values())
        self.last_round_pool = total_pool
        self.cumulative_pool += total_pool
        
        distributed_amount = (total_pool * self.multiplier) / self.num_of_agents
        rewards = {}
        for agent_id, contribution in actions.items():
            rewards[agent_id] = (self.initial_endowment - contribution) + distributed_amount
        return rewards

    