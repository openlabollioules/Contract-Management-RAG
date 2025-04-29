from typing import Optional
import time
import datetime

import ollama


class OllamaChat:
    def __init__(
        self,
        model: str = "mistral-small3.1:latest",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Ollama chat with a specific model and system prompt

        Args:
            model: The model to use (default: mistral-small3.1:latest)
            system_prompt: Optional system prompt to set the behavior of the model
        """
        self.model = model
        self.system_prompt = system_prompt
        self.messages = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, prompt: str, stream: bool = False) -> str:
        """
        Generate a response using Ollama chat

        Args:
            prompt: The input prompt
            stream: Whether to stream the response (default: False)

        Returns:
            The generated response
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        print(f"\n[OllamaChat.chat] [{datetime.datetime.now().strftime('%H:%M:%S')}] Calling model '{self.model}' with prompt length {len(prompt)}")
        response = ollama.chat(model=self.model, messages=self.messages, stream=stream)
        elapsed = time.time() - start_time
        print(f"[OllamaChat.chat] [{datetime.datetime.now().strftime('%H:%M:%S')}] Model '{self.model}' responded in {elapsed:.2f}s.")

        # Add assistant response to history
        self.messages.append(
            {"role": "assistant", "content": response["message"]["content"]}
        )

        return response["message"]["content"]

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """
        Generate a response using Ollama generate

        Args:
            prompt: The input prompt
            stream: Whether to stream the response (default: False)
            **kwargs: Additional parameters for generation

        Returns:
            The generated response
        """
        start_time = time.time()
        print(f"\n[OllamaChat.generate] [{datetime.datetime.now().strftime('%H:%M:%S')}] Calling model '{self.model}' with prompt length {len(prompt)}")
        response = ollama.generate(
            model=self.model, prompt=prompt, stream=stream, **kwargs
        )
        elapsed = time.time() - start_time
        print(f"[OllamaChat.generate] [{datetime.datetime.now().strftime('%H:%M:%S')}] Model '{self.model}' responded in {elapsed:.2f}s.")
        return response["response"]

    def reset(self) -> None:
        """Reset the chat history"""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})


# Global instance for backward compatibility
_ollama_chat = OllamaChat()


def ask_ollama(prompt: str, model: str = "mistral-small3.1:latest") -> str:
    """
    Generate a response using Ollama LLM (backward compatibility)

    Args:
        prompt: The input prompt
        model: The model to use (default: mistral-small3.1:latest)

    Returns:
        The generated response
    """
    _ollama_chat.model = model
    return _ollama_chat.generate(prompt)
