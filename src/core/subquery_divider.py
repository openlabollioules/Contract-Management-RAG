from typing import List
from pydantic import BaseModel, Field
from ollama import chat

class Questions(BaseModel):
    questions: List[str] = Field(
        description="Une liste de sous-questions liées à la requête d'entrée."
    )

def define_subqueries(query: str) -> List[str] :
    system_prompt = (
    "Tu es un assistant utile qui génère plusieurs sous-questions liées à une question d'entrée. "
    "L'objectif est de décomposer la question en un ensemble de sous-problèmes ou sous-questions pouvant être résolus indépendamment."
    )

    response = chat(
    model='mistral-small3.1',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ],
    format=Questions.model_json_schema()
    )

    questions = Questions.model_validate_json(response['message']['content'])

    return questions
