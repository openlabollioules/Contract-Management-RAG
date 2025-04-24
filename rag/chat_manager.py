from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from utils.logger import setup_logger

from .chroma_manager import ChromaDBManager

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class ChatManager:
    def __init__(
        self,
        chroma_manager: ChromaDBManager,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Initialize ChatManager with ChromaDB and OpenAI configuration

        Args:
            chroma_manager: Instance of ChromaDBManager for semantic search
            openai_api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
        """
        self.chroma_manager = chroma_manager
        self.llm = ChatOpenAI(
            model_name=model_name, openai_api_key=openai_api_key, temperature=0.7
        )
        logger.info(f"ChatManager initialisé avec le modèle {model_name}")

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            """,
                ),
                ("human", "{question}"),
            ]
        )
        logger.debug("Template de prompt configuré")

        # Create the chain
        self.chain = (
            {"context": self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        logger.debug("Chaîne de traitement configurée")

    def _format_docs(self, docs: List[Dict]) -> str:
        """Format the documents for the prompt"""
        formatted_docs = "\n\n".join(doc["document"] for doc in docs)
        logger.debug(f"Documents formatés pour le prompt ({len(docs)} documents)")
        return formatted_docs

    def chat(self, query: str, n_results: int = 3) -> str:
        """
        Chat with the documents using semantic search

        Args:
            query: The user's question
            n_results: Number of relevant documents to retrieve

        Returns:
            The generated response
        """
        logger.info(f"Chat: '{query}' (n_results={n_results})")

        # Get relevant documents
        logger.debug("Recherche de documents pertinents...")
        relevant_docs = self.chroma_manager.search(query, n_results=n_results)
        logger.debug(f"Documents trouvés: {len(relevant_docs)}")

        # Generate response
        logger.debug("Génération de la réponse...")
        response = self.chain.invoke(relevant_docs, query)
        logger.info(f"Réponse générée (longueur: {len(response)} caractères)")

        return response
