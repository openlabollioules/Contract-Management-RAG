from typing import List
from pydantic import BaseModel, Field
from ollama import chat


class Queries(BaseModel):
    """Schema for alternative search queries."""
    questions: List[str] = Field(
        description="A list of alternative search queries related to the original question."
    )


def generate_queries(question: str, model_name: str = 'mistral-small3.1') -> List[str]:
    """Generate multiple alternative queries for a given question.
    
    Args:
        question: The original question to generate alternatives for
        model_name: The Ollama model to use (default: mistral-small3.1)
        
    Returns:
        A list of alternative query strings
    """
    # Define system prompt
    system_prompt = (
        "You are an AI language model assistant. Your task is to generate five "
        "different versions of the given user question to retrieve relevant documents from a vector "
        "database. By generating multiple perspectives on the user question, your goal is to help "
        "the user overcome some of the limitations of the distance-based similarity search. "
        "Provide these alternative questions in your response."
    )

    # Call Ollama with the output schema
    response = chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ],
        format=Queries.model_json_schema()
    )

    # Validate and return the response
    queries = Queries.model_validate_json(response['message']['content'])
    return queries.questions


# Test the function
if __name__ == "__main__":
    original_question = "Quelles obligations du contrat A doivent être impérativement intégrées aux contrats qu'ALSTOM signera avec ses fournisseurs ou sous-traitants ?"
    alternative_queries = generate_queries(original_question)
    
    print(f"Original question: {original_question}")
    print("\nAlternative queries:")
    for i, query in enumerate(alternative_queries, 1):
        print(f"{i}. {query}") 