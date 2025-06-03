from typing import List, Dict, Any
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from ollama import chat
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class MultiQueryRetriever:
    """Retriever using multiple queries to retrieve documents."""

    def __init__(self, retriever: BaseRetriever, parser_key: str = "questions", debug: bool = False):
        """Initialize the MultiQueryRetriever.
        
        Args:
            retriever: Base retriever to use for document retrieval
            parser_key: Key to use for parsed output (default: "questions")
            debug: Whether to print debug information
        """
        self.retriever = retriever
        self.parser_key = parser_key
        self.debug = debug

    def _generate_queries(self, question: str) -> List[str]:
        """Generate multiple queries based on the original question."""
        # Define output schema
        class Queries(BaseModel):
            questions: List[str] = Field(
                description="A list of alternative search queries related to the original question."
            )

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
            model='mistral-small3.1',  # You can change the model as needed
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question}
            ],
            format=Queries.model_json_schema()
        )

        # Validate and return the response
        queries = Queries.model_validate_json(response['message']['content'])
        return queries.questions

    def invoke(self, question: str) -> List[Any]:
        """Get documents relevant to the query."""
        # Generate alternative queries
        queries = self._generate_queries(question)
        
        if self.debug:
            print("Alternative queries generated:")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
        
        # Include the original query
        all_queries = [question] + queries
        
        # Get documents for each query
        docs = []
        doc_ids = set()  # Track unique document IDs
        
        for query in all_queries:
            if self.debug:
                print(f"\nFetching documents for query: '{query}'")
            
            query_docs = self.retriever.invoke(query)
            
            if self.debug:
                print(f"  Found {len(query_docs)} documents")
            
            for doc in query_docs:
                # Only add unique documents
                doc_id = hash(doc.page_content)
                if doc_id not in doc_ids:
                    docs.append(doc)
                    doc_ids.add(doc_id)
                    if self.debug:
                        print(f"  Added new document: {doc.page_content[:50]}...")
        
        return docs


# Example usage
if __name__ == "__main__":
    from langchain_chroma import Chroma
    import os
    import sys
    
    try:
        # Essayer d'utiliser l'embeddings de text_vectorizer.py pour assurer la compatibilité
        sys.path.append('./src')  # Ajouter le dossier src au path
        try:
            from document_processing.text_vectorizer import TextVectorizer
            print("Utilisation de TextVectorizer du projet pour assurer la compatibilité")
            
            # Créer une classe wrapper pour adapter TextVectorizer à l'interface Embeddings de LangChain
            class TextVectorizerEmbeddings(Embeddings):
                def __init__(self):
                    self.vectorizer = TextVectorizer()
                
                def embed_documents(self, texts):
                    # get_embeddings retourne une liste d'embeddings numpy
                    embeddings = self.vectorizer.get_embeddings(texts)
                    # Convertir en liste pour LangChain
                    return [emb.tolist() for emb in embeddings]
                
                def embed_query(self, text):
                    # get_embeddings retourne une liste d'embeddings numpy
                    embeddings = self.vectorizer.get_embeddings([text])
                    # Retourner le premier (et unique) embedding comme liste
                    return embeddings[0].tolist()
            
            # Utiliser notre wrapper
            embeddings = TextVectorizerEmbeddings()
            
        except (ImportError, Exception) as e:
            print(f"Erreur lors de l'utilisation de TextVectorizer: {e}")
            # Fallback à une implémentation directe avec sentence_transformers
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            class CustomEmbeddings(Embeddings):
                def __init__(self, model_name):
                    print(f"Initialisation des embeddings personnalisés avec {model_name}")
                    self.model = SentenceTransformer(model_name, trust_remote_code=True)
                
                def embed_documents(self, texts):
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode(text, convert_to_numpy=True)
                    return embedding.tolist()
            
            embedding_model_name = "nomic-ai/nomic-embed-text-v2-moe"
            print(f"Utilisation du modèle d'embedding: {embedding_model_name}")
            embeddings = CustomEmbeddings(embedding_model_name)
        
        # Vérifier si le répertoire existe
        persist_directory = "./chroma_db"
        if not os.path.exists(persist_directory):
            print(f"ERREUR: Le répertoire {persist_directory} n'existe pas.")
            print("Veuillez vérifier le chemin de votre base Chroma.")
            exit(1)
        
        # Charger la base Chroma existante avec la collection spécifique 'contracts'
        existing_vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="contracts"  # Spécifier la collection contracts
        )
        
        # Afficher des statistiques sur la base
        collection = existing_vectordb._collection
        print(f"Base Chroma chargée avec succès.")
        print(f"Collection 'contracts' chargée avec succès.")
        print(f"Nombre d'éléments dans la collection: {collection.count()}")
        
    except Exception as e:
        print(f"ERREUR lors du chargement de la base Chroma: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Créer un retriever à partir de votre Chroma DB existante
    existing_retriever = existing_vectordb.as_retriever(
        search_type="similarity",  # ou "mmr" selon votre préférence
        search_kwargs={"k": 4}  # nombre de documents à récupérer par requête
    )

    # Créer le multi-query retriever avec votre retriever existant
    multi_query_retriever = MultiQueryRetriever(
        retriever=existing_retriever,
        parser_key="questions",
        debug=True  # Activer le mode debug
    )

    # Utiliser le retriever avec votre base existante
    question = "Quelle est la puissance délivrée attendue telle que spécifiée dans le contrat A ?"
    results = multi_query_retriever.invoke(question)
    
    # Afficher les résultats
    print(f"\nQuestion originale: {question}")
    print(f"Nombre de documents uniques trouvés: {len(results)}")
    
    # Afficher le contenu des documents trouvés
    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print("-" * 50)
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("-" * 50)
        if hasattr(doc, 'metadata') and doc.metadata:
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-" * 50) 