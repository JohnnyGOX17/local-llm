import requests
import sys


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model: str, prompt: str, system: str = "") -> str:
        """Generate response using Ollama"""
        url = f"{self.base_url}/api/generate"

        payload = {"model": model, "prompt": prompt, "stream": False}

        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"

    def list_models(self) -> list[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except:
            return []


if __name__ == "__main__":
    # Check if Ollama is running
    ollama_client = OllamaClient()
    models = ollama_client.list_models()

    if not models:
        print("⚠️  Ollama doesn't seem to be running or no models available")
        print("Make sure Ollama is installed and running: https://ollama.ai")
        sys.exit(1)

    for model in models:
        print(f"  - {model}")
