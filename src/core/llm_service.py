import os
import logging
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__file__)

class LLMService:
    def __init__(self, model_name: str = "mistral"):
        """
        Initialize LLM service for text analysis

        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLM model"""
        try:
            import ollama
            self.client = ollama.Client(host=os.getenv('OLLAMA_URL', 'http://localhost:11434'))
            logger.info(f"LLM service initialized with model: {self.model_name}")
        except ImportError:
            logger.warning("Ollama not installed. LLM features will be disabled.")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            self.client = None

    def generate(self, prompt: str, max_tokens: Optional[int] = 1000) -> str:
        """
        Generate text using the LLM

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response or empty string if LLM is not available
        """
        if not self.client:
            logger.warning("LLM service not available. Using fallback methods.")
            return ""
            
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error generating text with LLM: {str(e)}")
            return "" 