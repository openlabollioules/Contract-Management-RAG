import ollama
import logging

# Setup logger
logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, model: str = "mistral:latest"):
        """
        Initialize Ollama manager with a specific model

        Args:
            model: Name of the Ollama model to use
        """
        self.model = model

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Ollama model

        Args:
            prompt: The prompt to send to the model

        Returns:
            The generated response
        """
        try:
            response = ollama.generate(model=self.model, prompt=prompt, stream=False)
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Désolé, je n'ai pas pu générer de réponse."


def get_ollama_response(prompt: str, model: str = "mistral:latest") -> str:
    """
    Generate a response using Ollama without creating a manager instance

    Args:
        prompt: The prompt to send to the model
        model: Ollama model to use

    Returns:
        The generated response
    """
    try:
        response = ollama.generate(model=model, prompt=prompt, stream=False)
        return response["response"]
    except Exception as e:
        logger.error(f"Error generating response with model {model}: {str(e)}")
        return "Error generating response."
