import ollama


def ask_ollama(prompt: str, model: str = "mistral-small3.1:latest") -> str:
    """
    Generate a response using Ollama LLM

    Args:
        prompt: The input prompt
        model: The model to use (default: mistral-small3.1:latest)

    Returns:
        The generated response
    """
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']
