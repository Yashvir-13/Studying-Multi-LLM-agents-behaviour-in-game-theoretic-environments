from .llm_client import OllamaClient
import json

class BaseAgent():
    """
    Base LLM-driven agent for multi-agent simulations.
    
    Uses neutral prompt framing ("resource allocation game") to avoid
    biasing the LLM with game-theoretic terminology like "public goods."
    
    Attributes:
        id (str): Unique identifier for the agent.
        client (OllamaClient): Interface to the LLM.
        memory (list): History of past observations and results.
        memory_limit (int): Maximum number of past rounds to retain in context.
    """
    def __init__(self, id, client, memory_limit=5):
        """
        Args:
            id (str): Agent ID (e.g., "Agent_0").
            client (object): LLM client wrapper.
            memory_limit (int): Rolling window size for memory (default: 5).
        """
        self.id = id
        self.client = client
        self.memory = []
        self.memory_limit = memory_limit

    def act(self, observation):
        """
        Decide contribution based on observation.
        Uses neutral prompt framing to avoid LLM training bias.
        """
        # Slice memory to keep only the last 'memory_limit' rounds
        recent_memory = self.memory[-self.memory_limit:] if self.memory_limit > 0 else []
        
        # NEUTRAL FRAMING: No mention of "public goods game" or game theory terms
        system_prompt = (
            f"You are Participant {self.id} in a repeated resource allocation game. "
            "Each round, you receive tokens and choose how many to place into a shared pool. "
            "The pool is multiplied and split equally among all participants. "
            "Tokens you keep are yours. Your goal is to maximize your total earnings over all rounds. "
            "You must output ONLY valid JSON with a single integer field: "
            '{"contribution": X} where X is between 0 and the maximum tokens.'
        )
        
        user_prompt = (
            f"Game Configuration: {json.dumps(observation.get('config', {}))}\n"
            f"Your Memory (Last {len(recent_memory)} Rounds): {json.dumps(recent_memory)}\n"
            f"Current Round Observation: {json.dumps(observation)}\n"
            "How many tokens do you allocate to the shared pool? Output ONLY JSON."
        )
        
        try:
            response_json = self.client.generate(user_prompt, system_prompt=system_prompt)
            if not response_json:
                print(f"Agent {self.id} received empty response.")
                return {"contribution": 0}
                
            # cleanup json string if needed (sometimes models carry markdown)
            cleaned_json = response_json.replace("```json", "").replace("```", "").strip()
            action = json.loads(cleaned_json)
            return action
        except json.JSONDecodeError as e:
            print(f"Agent {self.id} failed to decode JSON: {response_json}. Error: {e}")
            return {"contribution": 0}
        except Exception as e:
            print(f"Agent {self.id} encountered error: {e}")
            return {"contribution": 0}

    def update(self, result):
        self.memory.append(result)
