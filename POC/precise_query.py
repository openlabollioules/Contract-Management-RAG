"""
self_query_poc.py - POC for implementing SelfQueryRetriever with contract management system
"""

import os
import sys
from typing import Dict, List, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.language_models.llms import LLM
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama as LangchainOllama

# Fix Python path to include the src directory
sys.path.append(os.path.abspath("."))

# Import Ollama for direct integration
import ollama
from dotenv import load_dotenv

# Ensure we're in the same environment as the rest of the application
load_dotenv("config.env")

# Configurez les journaux
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_query_poc")

# Debugging and analysis utilities
def visualize_self_query_filter(query_str, filter_str, limit_str=None):
    """
    Affiche la requête SQL générée par le SelfQueryRetriever dans un format plus lisible
    """
    import re
    
    # Format the components
    formatted = {
        "query": query_str.strip() if query_str else "(vide)",
        "filter": filter_str if filter_str else "aucun filtre",
        "limit": limit_str if limit_str else "aucune limite"
    }
    
    # Prettify the filter
    if "Comparison" in formatted["filter"]:
        # Extract field name
        field_match = re.search(r"attribute='([^']+)'", formatted["filter"])
        field = field_match.group(1) if field_match else "unknown"
        
        # Extract operator
        op_match = re.search(r"comparator=<Comparator\.([^:]+)", formatted["filter"])
        op = op_match.group(1) if op_match else "unknown"
        
        # Extract value
        val_match = re.search(r"value='([^']+)'", formatted["filter"])
        val = val_match.group(1) if val_match else "unknown"
        
        # Translate to SQL-like
        formatted["filter"] = f"WHERE {field} {op} '{val}'"
    
    # Format the output as SQL-like query
    sql_like = f"SELECT documents FROM collection\n{formatted['filter']}"
    if formatted["limit"] != "aucune limite":
        sql_like += f"\nLIMIT {formatted['limit']}"
    
    if formatted["query"] != "(vide)":
        sql_like += f"\nWITH SEMANTIC_SEARCH('{formatted['query']}')"
    
    return {
        "components": formatted,
        "sql_like": sql_like
    }

# Create a simple LLM wrapper for Ollama
class OllamaLLM(LLM):
    """Direct wrapper for Ollama LLM without relying on the imported module"""
    
    def __init__(self, model: str = None):
        """Initialize with the specified model"""
        super().__init__()
        # Use model from config.env if none specified
        if model is None:
            model = os.getenv("LLM_MODEL", "mistral-small3.1:latest")
        self.model = model
        logger.info(f"OllamaLLM initialized with model: {model}")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "ollama"
        
    def _call(self, prompt: str, **kwargs) -> str:
        """Call Ollama directly with the specified prompt"""
        try:
            logger.debug(f"Calling Ollama with prompt: {prompt[:50]}...")
            response = ollama.generate(model=self.model, prompt=prompt)
            return response["response"]
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error calling Ollama: {e}"

# Embeddings manager for the POC
class EmbeddingsManager:
    """Simplified embeddings manager for the POC"""
    
    def __init__(self, model_name: str = None):
        """Initialize with a sentence transformer model"""
        # Use model from config.env if none specified
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.model_name = model_name
        logger.info(f"Initializing embeddings model: {model_name}")
        
        try:
            # Pour le modèle nomic-ai/nomic-embed-text-v2-moe, nous devons ajouter trust_remote_code=True
            if "nomic" in model_name.lower():
                logger.info(f"Chargement du modèle Nomic avec trust_remote_code=True")
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_name)
                
            logger.info(f"Modèle d'embeddings chargé: {model_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            logger.info(f"Utilisation du modèle de fallback: BAAI/bge-m3")
            
            try:
                # Fallback to a reliable model
                self.model_name = "BAAI/bge-m3"
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Modèle de fallback chargé: {self.model_name}")
            except Exception as fallback_error:
                logger.error(f"Erreur lors du chargement du modèle de fallback: {fallback_error}")
                raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode(text).tolist()

def connect_to_chroma(db_path: str = "chroma_db", collection_name: str = None) -> Tuple[chromadb.Collection, Chroma]:
    """
    Connect to the existing ChromaDB and create a LangChain Chroma wrapper
    
    Returns a tuple of (chromadb_collection, langchain_chroma)
    """
    # Connect to the database
    logger.info(f"Connexion à la base de données existante dans {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    # Get the collections
    collections = client.list_collections()
    if not collections:
        logger.error(f"No collections found in {db_path}")
        raise ValueError(f"No collections found in {db_path}")
    
    # Extract collection names
    collection_names = [collection.name for collection in collections]
    logger.info(f"Collections disponibles: {collection_names}")
    
    # Use the first collection if none specified
    if collection_name is None or collection_name not in collection_names:
        collection_name = collection_names[0]
        logger.info(f"Utilisation de la collection: {collection_name}")
    
    # Get the collection
    native_collection = client.get_collection(name=collection_name)
    
    # Utiliser un modèle d'embeddings réel au lieu de vecteurs nuls
    embeddings_manager = EmbeddingsManager()
    
    # Create LangChain Chroma wrapper
    langchain_chroma = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings_manager
    )
    
    return native_collection, langchain_chroma

def get_metadata_schema(collection: chromadb.Collection) -> List[AttributeInfo]:
    """
    Analyze the collection to extract metadata schema
    
    Returns a list of AttributeInfo objects for SelfQueryRetriever
    """
    # Get a sample of documents to analyze metadata
    sample = collection.peek(limit=100)
    
    # Extract all metadata fields and their types
    metadata_fields = {}
    
    if not sample['metadatas']:
        logger.warning("No metadata found in collection")
        return []
    
    # Process each document's metadata
    for metadata in sample['metadatas']:
        for key, value in metadata.items():
            # Skip empty values
            if value is None or value == "":
                continue
                
            # Determine data type
            if key not in metadata_fields:
                if isinstance(value, bool) or (isinstance(value, str) and value.lower() in ['true', 'false']):
                    metadata_fields[key] = {'type': 'boolean', 'sample': value}
                elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    metadata_fields[key] = {'type': 'number', 'sample': value}
                else:
                    metadata_fields[key] = {'type': 'string', 'sample': value}
    
    # Descriptions plus précises pour les métadonnées
    field_descriptions = {
        "document_title": "Le titre exact du document contractuel (ex: 'Contrat_A_ALSTOM_oriented.pdf')",
        "dates": "Dates importantes mentionnées dans cette section du contrat, séparées par des points-virgules",
        "clause_type": "Le type de clause juridique (confidentialité, résiliation, paiement, etc.)",
        "keywords": "Mots-clés ou termes importants extraits de cette section",
        "has_cross_references": "Indique si cette section contient des références à d'autres parties du contrat",
        "chapter_title": "Le titre du chapitre auquel appartient cette section",
        "section_number": "Le numéro ou l'identifiant de la section dans le document",
        "content": "Le contenu textuel complet de cette section du contrat",
        "parent_section": "La section parente dont dépend cette section",
        "position": "La position de cette section dans le document",
        "total_chunks": "Le nombre total de sections dans le document"
    }
    
    # Convert to AttributeInfo objects with improved descriptions
    schema = []
    for field_name, field_info in metadata_fields.items():
        # Utiliser la description personnalisée ou une description générique
        description = field_descriptions.get(field_name, f"Information sur {field_name}")
        
        schema.append(
            AttributeInfo(
                name=field_name,
                description=description,
                type=field_info['type']
            )
        )
    
    return schema

# Étendre SelfQueryRetriever pour ajouter des fonctionnalités supplémentaires
class EnhancedSelfQueryRetriever(SelfQueryRetriever):
    """Version améliorée du SelfQueryRetriever avec capacités d'analyse et de débogage"""
    
    def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs):
        """
        Surcharge de la méthode interne pour capturer et analyser les requêtes générées
        
        Args:
            query: La requête utilisateur
            run_manager: Gestionnaire d'exécution fourni par LangChain
            **kwargs: Arguments supplémentaires
            
        Returns:
            Liste de documents pertinents
        """
        # Capture SelfQueryRetriever's logs to extract the query information
        import io
        import logging
        from langchain.retrievers.self_query.base import logger as self_query_logger
        
        # Create a string buffer to capture log output
        log_capture_string = io.StringIO()
        log_handler = logging.StreamHandler(log_capture_string)
        log_handler.setLevel(logging.INFO)
        self_query_logger.addHandler(log_handler)
        
        # Call the parent method (using the internal implementation to avoid recursion)
        docs = super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)
        
        # Get captured log
        log_contents = log_capture_string.getvalue()
        self_query_logger.removeHandler(log_handler)
        
        # Extract query components
        import re
        match = re.search(r"Generated Query: query='([^']*)'( filter=([^ ]*))?( limit=(\d+))?", log_contents)
        
        if match:
            query_parts = {
                "query": match.group(1) if match.group(1) else "",
                "filter": match.group(3) if match.group(3) else None,
                "limit": match.group(5) if match.group(5) else None
            }
            
            # Generate SQL-like visualization
            sql_info = visualize_self_query_filter(
                query_parts["query"], 
                query_parts["filter"], 
                query_parts["limit"]
            )
            
            # Log the SQL-like query
            logger.info(f"Query translated to SQL-like format:")
            logger.info(f"\n{sql_info['sql_like']}")
            
            # Enregistrer ces informations dans le run_manager si disponible
            if run_manager:
                try:
                    run_manager.on_text("Query Analysis", f"SQL-like format:\n{sql_info['sql_like']}")
                except Exception as e:
                    logger.debug(f"Couldn't log to run_manager: {e}")
        
        return docs

def create_self_query_retriever(
    vectorstore: Chroma, 
    metadata_field_info: List[AttributeInfo],
    model_name: str = None
) -> EnhancedSelfQueryRetriever:
    """
    Create a SelfQueryRetriever instance using Ollama
    
    Args:
        vectorstore: The LangChain Chroma wrapper
        metadata_field_info: List of AttributeInfo objects for metadata schema
        model_name: Ollama model to use (default: use from config.env)
    
    Returns:
        SelfQueryRetriever instance
    """
    # Use model from config.env if none specified
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "mistral-small3.1")
        
    # Use LangChain's built-in Ollama integration instead of custom implementation
    logger.info(f"Initializing Ollama with model: {model_name}")
    try:
        llm = LangchainOllama(model=model_name)
    except Exception as e:
        logger.error(f"Error initializing LangchainOllama: {e}")
        # Fallback to our custom implementation
        llm = OllamaLLM(model=model_name)
    
    # Améliorer la description du contexte
    document_content_description = """
    Sections de documents contractuels juridiques incluant diverses clauses, définitions et termes légaux.
    Les documents sont structurés en sections et sous-sections numérotées, avec différents types de clauses 
    comme la confidentialité, le paiement, la résiliation, etc. Les noms de fichiers contiennent souvent 
    des underscores et des suffixes (ex: 'Contrat_A_ALSTOM_oriented.pdf').
    """
    
    # Créer le retriever avec des paramètres optimisés
    retriever = EnhancedSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=None,
        verbose=True,
        enable_limit=True  # Activer la limite pour contrôler le nombre de résultats
    )
    
    return retriever

def format_results(results: List[Document]) -> List[Dict]:
    """Format retriever results for display"""
    formatted = []
    for doc in results:
        formatted.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        })
    return formatted

def save_results_to_json(query: str, results: List[Document], filename: str = "query_results.json"):
    """Save query results to a JSON file for later analysis"""
    import json
    from datetime import datetime
    
    # Format the results
    formatted_results = []
    for i, doc in enumerate(results):
        formatted_results.append({
            "result_index": i + 1,
            "content": doc.page_content,
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        })
    
    # Create the output structure
    output = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "results": formatted_results
    }
    
    # Write to file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False, indent=2))
        f.write("\n---\n")  # Add separator for multiple queries
    
    logger.info(f"Results saved to {filename}")
    
    return filename

def post_process_results(query: str, results: List[Document], top_k: int = 5) -> List[Document]:
    """Post-traitement des résultats pour améliorer la pertinence"""
    
    # Si aucun résultat n'a été trouvé avec les filtres exacts, essayer une recherche plus souple
    if not results:
        logger.info("Aucun résultat trouvé avec les filtres exacts, essai avec une recherche plus souple")
        # Extraire les mots clés de la requête (simpliste, à améliorer)
        keywords = query.lower().split()
        keywords = [k for k in keywords if len(k) > 3]  # Ignore les mots courts
        
        # Filtre souple sur le contenu ou les métadonnées
        filtered_results = []
        
        # Cette partie dépendrait de l'accès direct à la collection
        # Pour un POC, nous pourrions ajouter cette logique plus tard
        
        return filtered_results[:top_k]
    
    # Trier les résultats par pertinence si nous en avons
    if results:
        # Tri par pertinence des métadonnées (exemple: prioriser les documents avec dates si la requête concerne des dates)
        if "date" in query.lower():
            results.sort(key=lambda doc: 1 if doc.metadata.get("dates") else 0, reverse=True)
            
        # Limiter le nombre de résultats
        results = results[:top_k]
    
    return results

# Questions de test avancées provenant de test_rag.py
TEST_QUESTIONS = [
    "Peux-tu m'indiquer les dates clés du Contrat A ?",
    "Peux-tu me lister les éléments du contrat A qui impliquent le paiement potentiel d'indemnités ou de pénalités de la part du fournisseur ?",
    "Peux-tu résumer les obligations de garantie prévues dans le contrat A ?",
    "Dans le contrat A, quelle est la clause qui est la plus problématique du point de vue du fournisseur et pourquoi ?",
    "Dans le contrat A, quel est le risque de change introduit par le fait qu'une partie des prix soient établis en roubles ?",
    "Quelle est la puissance délivrée attendue telle que spécifiée dans le contrat A ?",
    "Quelles sont les lois applicables mentionnées dans le contrat A ?",
    "Je suis le représentant du fournisseur. J'aimerais envoyer un Courier de notification de retard au client du contrat A concernant des retards subis de sa part.",
    "A partir du contrat A, peux-tu dresser la liste des actions à mener par le fournisseur en termes de documents à fournir au client ?",
    "Quelles obligations du contrat A doivent être impérativement intégrées aux contrats qu'ALSTOM signera avec ses fournisseurs ou sous-traitants ?",
    "Comment traduire la clause de garantie du contrat A vis-à-vis des fournisseurs et sous-traitants de Alstom ?",
    "Comment traduire la clause de responsabilité du contrat A vis-à-vis des fournisseurs et sous-traitants de Alstom ?"
]

def main():
    """Run the POC"""
    try:
        # Connect to the database
        logger.info("Connexion à la base ChromaDB existante...")
        collection, vectorstore = connect_to_chroma(db_path="chroma_db-1")
        
        # Get metadata schema
        logger.info("Analyse du schéma de métadonnées...")
        metadata_field_info = get_metadata_schema(collection)
        logger.info(f"Trouvé {len(metadata_field_info)} champs de métadonnées:")
        for field in metadata_field_info:
            logger.info(f"  - {field.name} ({field.type}): {field.description}")
        
        # Create self-query retriever with Ollama
        logger.info("Creating SelfQueryRetriever with Ollama...")
        retriever = create_self_query_retriever(vectorstore, metadata_field_info)
        
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Test SelfQueryRetriever with contract questions")
        parser.add_argument("--query", type=str, help="Specific query to test")
        parser.add_argument("--all", action="store_true", help="Test all questions from TEST_QUESTIONS")
        parser.add_argument("--limit", type=int, default=5, help="Limit number of results")
        parser.add_argument("--output", type=str, default="query_results.json", help="Output file for results")
        args = parser.parse_args()
        
        # If a specific query is requested
        if args.query:
            logger.info(f"Testing with query: '{args.query}'")
            results = retriever.get_relevant_documents(args.query)
            
            # Show results
            logger.info(f"Retrieved {len(results)} documents")
            formatted_results = format_results(results)
            for i, result in enumerate(formatted_results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Content: {result['content']}")
                logger.info(f"  Metadata: {result['metadata']}")
                logger.info("-" * 50)
                
            # Save results to file
            save_results_to_json(args.query, results, args.output)
        
        # If --all flag is set, test all questions
        elif args.all:
            logger.info("Testing all questions from TEST_QUESTIONS")
            all_results = {}
            
            for query in TEST_QUESTIONS:
                logger.info(f"\nTesting question: '{query}'")
                results = retriever.get_relevant_documents(query)
                
                # Post-process results to improve relevance
                results = post_process_results(query, results, top_k=args.limit)
                
                logger.info(f"Retrieved {len(results)} documents")
                if results:
                    content_preview = results[0].page_content[:100] + "..." if len(results[0].page_content) > 100 else results[0].page_content
                    logger.info(f"First result: {content_preview}")
                    logger.info(f"Metadata: {results[0].metadata}")
                
                # Save each query's results
                save_results_to_json(query, results, args.output)
                
                # Store for summary
                all_results[query] = len(results)
            
            # Print summary
            logger.info("\n=== Summary of all queries ===")
            for query, num_results in all_results.items():
                logger.info(f"Query: '{query}' - {num_results} results")
        
        # Default behavior: test with basic queries
        else:
            # Test with the sample query from rewrite_query.py
            test_query = "Peux-tu m'indiquer les dates clés du Contrat A ?"
            logger.info(f"Testing with query: '{test_query}'")
            results = retriever.get_relevant_documents(test_query)
            
            # Show results
            logger.info(f"Retrieved {len(results)} documents")
            formatted_results = format_results(results)
            for i, result in enumerate(formatted_results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Content: {result['content']}")
                logger.info(f"  Metadata: {result['metadata']}")
                logger.info("-" * 50)
            
            # Save results to file
            save_results_to_json(test_query, results, args.output)
            
            # Test with metadata-focused queries
            metadata_queries = [
                "Montre-moi les clauses de confidentialité du document Contrat_A_ALSTOM",
                "Quelles sont les clauses de résiliation mentionnant une date spécifique?",
                "Trouve les sections avec des références croisées qui parlent de paiement",
                "Sections du contrat A qui mentionnent le mot 'paiement' dans leur contenu"
            ]
            
            for query in metadata_queries:
                logger.info(f"\nTesting metadata query: '{query}'")
                results = retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(results)} documents")
                if results:
                    logger.info(f"First result metadata: {results[0].metadata}")
                
                # Save each query's results
                save_results_to_json(query, results, args.output)
    
    except Exception as e:
        logger.error(f"Error running POC: {e}", exc_info=True)

if __name__ == "__main__":
    main()