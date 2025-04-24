import ollama

from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class OllamaManager:
    def __init__(self, model: str = "mistral:latest"):
        """
        Initialize Ollama manager with a specific model

        Args:
            model: Name of the Ollama model to use
        """
        self.model = model
        logger.info(f"OllamaManager initialisé avec le modèle: {model}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Ollama model

        Args:
            prompt: The prompt to send to the model

        Returns:
            The generated response
        """
        logger.info(f"Génération de réponse avec le modèle: {self.model}")
        logger.debug(f"Prompt: {prompt[:50]}...")

        try:
            logger.debug("Appel à ollama.generate...")
            response = ollama.generate(model=self.model, prompt=prompt, stream=False)
            logger.info(
                f"Réponse générée (longueur: {len(response['response'])} caractères)"
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            print(f"Erreur lors de la génération de la réponse: {str(e)}")
            return "Désolé, je n'ai pas pu générer de réponse."
