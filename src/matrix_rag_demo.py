#!/usr/bin/env python3
"""
matrix_rag_demo.py - Démonstration du RAG amélioré avec Matrices G et Doc
"""

import os
import sys
import argparse
from pathlib import Path
import chromadb
from loguru import logger

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer les modules nécessaires
from src.core.matrix_processor import MatrixProcessor
from src.core.enhanced_retriever import create_enhanced_retriever
from src.utils.logger import init_logger
from src.document_processing.llm_chat import ask_ollama  # Adapter selon votre implémentation

# Chemin vers les données
DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))

def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Démonstration du RAG amélioré avec Matrices G et Doc")
    
    parser.add_argument(
        "--matrix-g", 
        default=str(DATA_DIR / "matrix_g.csv"),
        help="Chemin vers le fichier de la Matrice G"
    )
    
    parser.add_argument(
        "--matrix-doc", 
        default=str(DATA_DIR / "matrix_doc.csv"),
        help="Chemin vers le fichier de la Matrice Doc"
    )
    
    parser.add_argument(
        "--chroma-path", 
        default="./chroma_db",
        help="Chemin vers la base de données Chroma"
    )
    
    parser.add_argument(
        "--collection-name",
        default="default",
        help="Nom de la collection Chroma à utiliser"
    )
    
    parser.add_argument(
        "--langchain",
        action="store_true",
        help="Utiliser l'implémentation LangChain"
    )
    
    parser.add_argument(
        "--query",
        help="Requête à exécuter"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mode interactif pour tester plusieurs requêtes"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de résultats à retourner"
    )
    
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "mistral-large3:latest"),
        help="Modèle LLM à utiliser pour la génération de réponses"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.3")),
        help="Température pour la génération"
    )
    
    return parser.parse_args()

def format_sources_for_prompt(sources):
    """Formate les sources pour le prompt LLM"""
    formatted_sources = ""
    
    for i, source in enumerate(sources, 1):
        # Gérer différents formats de sources possibles
        if isinstance(source, dict):
            if "content" in source:
                content = source["content"]
            elif "page_content" in source:  # Format LangChain
                content = source["page_content"]
            else:
                content = str(source)
                
            # Récupérer les métadonnées
            metadata = source.get("metadata", {})
            
            # Formater la source
            formatted_sources += f"SOURCE {i}:\n"
            
            # Ajouter les métadonnées pertinentes
            if "idG" in metadata:
                formatted_sources += f"IdG: {metadata['idG']}\n"
            if "titre" in metadata:
                formatted_sources += f"Titre: {metadata['titre']}\n"
            if "type" in metadata:
                formatted_sources += f"Type: {metadata['type']}\n"
            
            # Ajouter le contenu
            formatted_sources += f"Contenu: {content}\n\n"
        elif hasattr(source, "page_content"):  # Objet Document de LangChain
            content = source.page_content
            metadata = getattr(source, "metadata", {})
            
            # Formater la source
            formatted_sources += f"SOURCE {i}:\n"
            
            # Ajouter les métadonnées pertinentes
            if "idG" in metadata:
                formatted_sources += f"IdG: {metadata['idG']}\n"
            if "titre" in metadata:
                formatted_sources += f"Titre: {metadata['titre']}\n"
            if "type" in metadata:
                formatted_sources += f"Type: {metadata['type']}\n"
            
            # Ajouter le contenu
            formatted_sources += f"Contenu: {content}\n\n"
        else:
            # Fallback pour tout autre format
            formatted_sources += f"SOURCE {i}:\n{str(source)}\n\n"
    
    return formatted_sources

def generate_answer(query, sources, model, temperature):
    """Génère une réponse à partir des sources récupérées"""
    # Formater les sources pour le prompt
    formatted_sources = format_sources_for_prompt(sources)
    
    # Construire le prompt
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats juridiques. Utilise les sources fournies pour répondre à la question de l'utilisateur.

SOURCES:
{formatted_sources}

QUESTION: {query}

INSTRUCTIONS:
1. Réponds précisément à la question en te basant UNIQUEMENT sur les informations présentes dans les sources fournies
2. Si l'information n'est pas disponible dans les sources, indique clairement que "Cette information n'est pas disponible dans les sources"
3. Ne fais aucune supposition ou déduction au-delà des informations explicitement présentes dans les sources
4. Cite les identifiants des sources (IdG) pertinentes dans ta réponse
5. Organise ta réponse de manière claire et structurée
6. Sois précis et concis

RÉPONSE:"""
    
    # Appeler le LLM pour générer la réponse
    try:
        response = ask_ollama(prompt, temperature, model)
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {e}")
        return f"Erreur: {e}"

def create_chromadb_retriever(client, collection_name):
    """Crée un retriever basé sur ChromaDB"""
    # Récupérer ou créer la collection
    collection = client.get_or_create_collection(name=collection_name)
    
    # Créer un retriever simple
    def retrieve(query, top_k=5, filter=None):
        # Paramètres de la requête
        query_params = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Ajouter le filtre si présent
        if filter is not None:
            query_params["where"] = filter
        
        # Exécuter la requête
        results = collection.query(**query_params)
        
        # Formater les résultats
        formatted_results = []
        
        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    # Créer un objet retriever avec la fonction retrieve
    class ChromaRetriever:
        def retrieve(self, query, top_k=5, filter=None):
            return retrieve(query, top_k, filter)
    
    return ChromaRetriever()

def create_langchain_retriever(client, collection_name):
    """Crée un retriever basé sur LangChain"""
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.embeddings import Embeddings
        
        # Créer un objet d'embeddings minimal pour utiliser avec Chroma
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts):
                # On ne l'utilise pas vraiment, car les documents sont déjà indexés
                raise NotImplementedError("Cette méthode ne devrait pas être appelée")
            
            def embed_query(self, text):
                # ChromaDB effectue l'embedding côté serveur
                return [0.0] * 384  # Dimension quelconque
        
        # Créer le vectorstore LangChain
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=DummyEmbeddings()
        )
        
        return vector_store
    except ImportError:
        logger.error("LangChain n'est pas installé. Impossible de créer un retriever LangChain.")
        sys.exit(1)

def main():
    """Fonction principale"""
    # Initialiser le logger
    init_logger()
    
    # Parser les arguments
    args = parse_arguments()
    
    # Initialiser le processeur de matrices
    logger.info("Initialisation du processeur de matrices")
    matrix_processor = MatrixProcessor(
        matrix_g_path=args.matrix_g,
        matrix_doc_path=args.matrix_doc
    )
    
    # Initialiser le client ChromaDB
    logger.info(f"Connexion à ChromaDB à {args.chroma_path}")
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    
    # Créer le retriever de base
    if args.langchain:
        logger.info("Utilisation de l'implémentation LangChain")
        base_retriever = create_langchain_retriever(chroma_client, args.collection_name)
        retriever_type = "langchain"
    else:
        logger.info("Utilisation de l'implémentation ChromaDB standard")
        base_retriever = create_chromadb_retriever(chroma_client, args.collection_name)
        retriever_type = "default"
    
    # Créer le retriever amélioré
    enhanced_retriever = create_enhanced_retriever(
        retriever_type=retriever_type,
        vector_store=base_retriever,
        matrix_processor=matrix_processor,
        fallback_enabled=True
    )
    
    # Exécuter une requête unique ou entrer en mode interactif
    if args.query:
        # Exécuter une requête unique
        query = args.query
        logger.info(f"Exécution de la requête: {query}")
        
        # Récupérer les documents pertinents
        if retriever_type == "langchain":
            results = enhanced_retriever.get_relevant_documents(query, k=args.top_k)
        else:
            results = enhanced_retriever.retrieve(query, top_k=args.top_k)
        
        # Afficher les résultats
        print(f"\nRésultats pour la requête: {query}")
        print(f"Nombre de résultats: {len(results)}")
        
        # Générer une réponse
        answer = generate_answer(query, results, args.model, args.temperature)
        
        print("\n===== RÉPONSE =====")
        print(answer)
        print("===================\n")
        
        # Afficher les sources
        print("\n===== SOURCES =====")
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                # Format standard
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                print(f"\nSOURCE {i}:")
                print(f"IdG: {metadata.get('idG', 'N/A')}")
                print(f"Titre: {metadata.get('titre', 'N/A')}")
                print(f"Type: {metadata.get('type', 'N/A')}")
                print(f"Contenu: {content[:200]}...")
            elif hasattr(result, "page_content"):
                # Format LangChain
                content = result.page_content
                metadata = getattr(result, "metadata", {})
                print(f"\nSOURCE {i}:")
                print(f"IdG: {metadata.get('idG', 'N/A')}")
                print(f"Titre: {metadata.get('titre', 'N/A')}")
                print(f"Type: {metadata.get('type', 'N/A')}")
                print(f"Contenu: {content[:200]}...")
            else:
                print(f"\nSOURCE {i}: {str(result)[:200]}...")
        print("===================\n")
    
    elif args.interactive:
        # Mode interactif
        print("\n===== MODE INTERACTIF =====")
        print("Entrez vos requêtes (tapez 'exit' pour quitter)")
        
        while True:
            query = input("\nRequête > ")
            
            if query.lower() in ["exit", "quit", "q", "bye"]:
                break
            
            if not query.strip():
                continue
            
            # Récupérer les documents pertinents
            if retriever_type == "langchain":
                results = enhanced_retriever.get_relevant_documents(query, k=args.top_k)
            else:
                results = enhanced_retriever.retrieve(query, top_k=args.top_k)
            
            # Afficher le nombre de résultats
            print(f"Nombre de résultats: {len(results)}")
            
            # Générer une réponse
            answer = generate_answer(query, results, args.model, args.temperature)
            
            print("\n===== RÉPONSE =====")
            print(answer)
            print("===================\n")
            
            # Demander si l'utilisateur veut voir les sources
            show_sources = input("Afficher les sources? (o/n) > ")
            
            if show_sources.lower() in ["o", "oui", "y", "yes"]:
                print("\n===== SOURCES =====")
                for i, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        # Format standard
                        content = result.get("content", "")
                        metadata = result.get("metadata", {})
                        print(f"\nSOURCE {i}:")
                        print(f"IdG: {metadata.get('idG', 'N/A')}")
                        print(f"Titre: {metadata.get('titre', 'N/A')}")
                        print(f"Type: {metadata.get('type', 'N/A')}")
                        print(f"Contenu: {content[:200]}...")
                    elif hasattr(result, "page_content"):
                        # Format LangChain
                        content = result.page_content
                        metadata = getattr(result, "metadata", {})
                        print(f"\nSOURCE {i}:")
                        print(f"IdG: {metadata.get('idG', 'N/A')}")
                        print(f"Titre: {metadata.get('titre', 'N/A')}")
                        print(f"Type: {metadata.get('type', 'N/A')}")
                        print(f"Contenu: {content[:200]}...")
                    else:
                        print(f"\nSOURCE {i}: {str(result)[:200]}...")
                print("===================\n")
    
    else:
        print("Aucune action spécifiée. Utilisez --query ou --interactive pour tester le système.")
        print("Utilisez --help pour voir toutes les options disponibles.")

if __name__ == "__main__":
    main()