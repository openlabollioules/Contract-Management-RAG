import ollama


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
            print(f"Erreur lors de la génération de la réponse: {str(e)}")
            return "Désolé, je n'ai pas pu générer de réponse."
