from typing import List
from pydantic import BaseModel, Field
from ollama import chat

# Définir le schéma de sortie
class Questions(BaseModel):
    questions: List[str] = Field(
        description="Une liste de sous-questions liées à la requête d'entrée."
    )

# Définir le message système
system_prompt = (
    "Tu es un assistant utile qui génère plusieurs sous-questions liées à une question d'entrée. "
    "L'objectif est de décomposer la question en un ensemble de sous-problèmes ou sous-questions pouvant être résolus indépendamment."
)

# Définir la question utilisateur
user_question = "Quelles sont les lois applicables mentionnées dans le contrat A ?"

# Appeler le modèle Ollama avec le schéma de sortie
response = chat(
    model='mistral-small3.1',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question}
    ],
    format=Questions.model_json_schema()
)

# Valider et afficher la réponse
questions = Questions.model_validate_json(response['message']['content'])
print(questions)
