import re
import time
import ollama

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds; doubles each attempt

# Matches the first {...} JSON object in any messy text
_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def _extract_json(text: str) -> str:
    """
    Robustly extract the first JSON object from messy model output.

    Handles:
    - <think>...</think> reasoning blocks (DeepSeek-R1, QwQ)
    - Markdown code fences (```json ... ```)
    - Bold/italic headers (**Response:**, etc.)
    - Extra text before/after the JSON
    """
    if not text:
        return text

    # 1. Strip <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Extract content from JSON code fences if present
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Find first bare {...} object
    obj_match = _JSON_RE.search(text)
    if obj_match:
        return obj_match.group(0).strip()

    # 4. Fall back to the stripped text as-is (let the caller handle parse errors)
    return text.strip()


class OllamaClient:
    """
    LLM client backed by local Ollama server.
    Requires: ollama serve (running locally on port 11434)

    Works with both standard models (llama3) and reasoning models
    (deepseek-r1, qwq) that emit <think> blocks or markdown-wrapped JSON.
    """

    def __init__(self, model: str = "llama3", temperature: float = 0.7,
                 num_predict: int = 128, think=None):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.think = think

    def generate(self, message: str, system_prompt: str = None) -> str | None:
        """
        Send a chat completion request to Ollama.
        Returns a clean JSON string extracted from the response, or None on total failure.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                chat_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "options": {
                        "temperature": self.temperature,
                        # Keep output budgets tight because agents only need tiny JSON replies.
                        "num_predict": self.num_predict,
                        "format": "json",
                    },
                    "keep_alive": -1,
                }
                if self.think is not None:
                    chat_kwargs["think"] = self.think

                response = ollama.chat(**chat_kwargs)
                raw = response['message']['content']
                return _extract_json(raw)

            except Exception as e:
                last_error = e
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"[OllamaClient] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        print(f"[OllamaClient] All {MAX_RETRIES} attempts failed. Last error: {last_error}")
        return None


if __name__ == "__main__":
    client = OllamaClient()
    print(f"Using model: {client.model}")
    print("Enter your message (Ctrl-C to quit):")
    try:
        while True:
            msg = input("You: ")
            if msg.lower() in ("exit", "quit"):
                break
            print("LLM:", client.generate(msg))
    except KeyboardInterrupt:
        print("\nExiting...")
