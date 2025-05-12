import os
import platform
from typing import Optional

import ollama
import torch
from dotenv import load_dotenv

from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)

# Load environment variables
load_dotenv("config.env")

# Configuration pour Ollama
os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_URL", "http://localhost:11434").split(
    "/api"
)[0]
logger.info(f"OLLAMA_BASE_URL configuré à {os.environ['OLLAMA_BASE_URL']}")

# Configuration pour les modèles offline
USE_OFFLINE_MODELS = os.getenv("USE_OFFLINE_MODELS", "true").lower() == "true"
if USE_OFFLINE_MODELS:
    logger.info("Mode hors ligne activé, utilisation des modèles locaux")
    # Configurer les variables d'environnement pour le mode hors ligne
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Détection du matériel (pour optimisations)
is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
if is_apple_silicon and torch.backends.mps.is_available():
    device = "mps"
    logger.info("🎮 Using MPS (Metal Performance Shaders) for LLM")
elif torch.cuda.is_available():
    device = "cuda"
    logger.info("🚀 Using CUDA for LLM")
else:
    device = "cpu"
    logger.info("💻 Using CPU for LLM")


class LLMChat:
    def __init__(
        self,
        model: str = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Ollama chat with a specific model and system prompt

        Args:
            model: The model to use (default is taken from environment variable)
            system_prompt: Optional system prompt to set the behavior of the model
        """
        # Get model name from environment variables or use default
        if model is None:
            self.model = os.getenv("LLM_MODEL", "command-a:latest")
        else:
            self.model = model

        self.system_prompt = system_prompt
        self.messages = []
        logger.info(f"LLMChat initialisé avec le modèle {self.model}")

        if system_prompt:
            logger.debug(f"System prompt défini: {system_prompt[:50]}...")
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
        logger.info(f"Chat avec Ollama (modèle: {self.model}, stream: {stream})")
        logger.debug(f"Prompt: {prompt[:50]}...")

        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})
        logger.debug(f"Historique de messages: {len(self.messages)} messages")

        # Generate response
        logger.debug("Génération de la réponse...")
        response = ollama.chat(model=self.model, messages=self.messages, stream=stream)
        logger.debug("Réponse reçue")

        # Add assistant response to history
        self.messages.append(
            {"role": "assistant", "content": response["message"]["content"]}
        )
        logger.debug(
            f"Réponse ajoutée à l'historique (longueur totale: {len(self.messages)} messages)"
        )

        logger.info(
            f"Réponse générée (longueur: {len(response['message']['content'])} caractères)"
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
        logger.info(f"Génération avec Ollama (modèle: {self.model}, stream: {stream})")
        logger.debug(f"Prompt: {prompt[:50]}...")
        logger.debug(f"Paramètres supplémentaires: {kwargs}")

        logger.debug("Appel à ollama.generate...")
        response = ollama.generate(
            model=self.model, prompt=prompt, stream=stream, **kwargs, options={'temperature':0.7}
        )
        logger.debug("Réponse reçue")

        logger.info(
            f"Réponse générée (longueur: {len(response['response'])} caractères)"
        )
        return response["response"]

    def reset(self) -> None:
        """Reset the chat history"""
        logger.info("Réinitialisation de l'historique de chat")
        old_length = len(self.messages)
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        logger.debug(f"Historique réinitialisé ({old_length} messages supprimés)")


# Global instance for backward compatibility
_ollama_chat = LLMChat()
logger.debug("Instance globale _ollama_chat créée")


def ask_ollama(prompt: str, model: str = "mistral-small3.1:latest") -> str:
    """
    Generate a response using Ollama LLM (backward compatibility)

    Args:
        prompt: The input prompt
        model: The model to use (default: mistral-small3.1:latest)

    Returns:
        The generated response
    """
    logger.info(f"ask_ollama appelé avec modèle: {model}")
    logger.debug(f"Changement de modèle de l'instance globale à: {model}")
    _ollama_chat.model = model
    logger.debug(f"Génération avec prompt: {prompt[:50]}...")
    response = _ollama_chat.generate(prompt)
    logger.info(
        f"Réponse générée via ask_ollama (longueur: {len(response)} caractères)"
    )
    return response
