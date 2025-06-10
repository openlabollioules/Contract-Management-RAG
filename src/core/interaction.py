from document_processing.llm_chat import llm_chat_call_with_ollama
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from core.graph_manager import GraphManager
from document_processing.reranker import Reranker
from core.subquery_divider import define_subqueries
from core.alternative_query_creator import generate_queries
from utils.logger import setup_logger
from utils.hybrid_search import HybridSearch
import os
from dotenv import load_dotenv
# Import the query classification
from core.query_classification_v2 import determine_inference_mode
from core.history_func import retrieve_similar_conversations, add_conversation_in_history_db, get_history

# Configurer le logger pour ce module
logger = setup_logger(__file__)
load_dotenv("config.env")

# Variable globale pour stocker l'instance de recherche hybride
hybrid_search_instance = None

def get_singleton_hybrid_search(embeddings_manager, chroma_manager, alpha=0.7):
    """
    Récupère ou initialise l'instance de recherche hybride
    
    Args:
        embeddings_manager: Instance du gestionnaire d'embeddings
        chroma_manager: Instance du gestionnaire ChromaDB
        alpha: Poids pour les scores vectoriels (1-alpha pour BM25)
    
    Returns:
        HybridSearch: Instance de recherche hybride
    """
    global hybrid_search_instance
    
    if hybrid_search_instance is None:
        logger.info("Initialisation de la recherche hybride (vectorielle + BM25)...")
        hybrid_search_instance = HybridSearch(chroma_manager, alpha=alpha)
        
        # Récupérer tous les documents de ChromaDB pour entraîner BM25
        logger.info("Récupération des documents pour l'indexation BM25...")
        all_docs = chroma_manager.get_all_documents_with_content()
        
        if all_docs:
            logger.info(f"Indexation de {len(all_docs)} documents avec BM25...")
            hybrid_search_instance.fit(all_docs)
        else:
            logger.warning("Aucun document disponible pour l'indexation BM25")
    
    return hybrid_search_instance

def display_contract_search_results(query: str, n_results: int = 5, use_hybrid: bool = True) -> None:
    """
    Search in the contract database

    Args:
        query: Search query
        n_results: Number of results to return
        use_hybrid: Whether to use hybrid search (BM25 + semantic)
    """
    logger.info(f"\n🔍 Recherche: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)
    
    # Choose search method
    if use_hybrid:
        logger.info("Utilisation de la recherche hybride (vectorielle + BM25)...")
        hybrid_search = get_singleton_hybrid_search(embeddings_manager, chroma_manager)
        results = hybrid_search.search(query, n_results=n_results)
    else:
        # Standard vector search
        logger.info("Utilisation de la recherche vectorielle standard...")
        results = chroma_manager.search(query, n_results=n_results)

    # Display results
    logger.info(f"\n📊 Résultats ({len(results)} trouvés):")
    for i, result in enumerate(results, 1):
        logger.info(f"\n--- Résultat {i} ---")
        logger.info(
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}"
        )
        logger.info(
            f"Hiérarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}"
        )
        logger.info(
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}"
        )
        logger.info(f"Contenu: {result['document'][:200]}...")
        logger.info(f"Distance vectorielle: {result['distance']:.4f}")
        
        # Afficher les scores BM25 si disponibles
        if 'bm25_score' in result:
            logger.info(f"Score BM25: {result['bm25_score']:.4f}")
            logger.info(f"Score combiné: {result['combined_score']:.4f}")

def load_or_recreate_knowledge_graph(chroma_manager, embeddings_manager):
    """
    Load existing graph or build a new one if needed
    
    Args:
        chroma_manager: ChromaDB manager instance
        embeddings_manager: Embeddings manager instance
    
    Returns:
        KnowledgeGraph: The loaded or built knowledge graph
    """
    import os
    import pickle
    
    graph_path = "knowledge_graph.pkl"
    
    # Check if graph file exists
    if os.path.exists(graph_path):
        try:
            logger.info(f"📂 Chargement du graphe existant depuis {graph_path}...")
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du graphe: {str(e)}")
    
    # If no graph exists or loading failed, build a new one
    logger.info("🔄 Construction d'un nouveau graphe de connaissances...")
    
    # Get all documents from ChromaDB
    docs_dict = chroma_manager.get_all_documents()
    
    # Convert the dictionary to the expected format for build_graph
    all_docs = []
    
    # We need the actual content, so fetch the documents with their content
    all_docs_with_content = chroma_manager.collection.get()
    
    # Create a mapping of id -> content
    id_to_content = {}
    if "ids" in all_docs_with_content and "documents" in all_docs_with_content:
        for i, doc_id in enumerate(all_docs_with_content["ids"]):
            if i < len(all_docs_with_content["documents"]):
                id_to_content[doc_id] = all_docs_with_content["documents"][i]
    
    # Now create the list of dicts expected by build_graph
    for doc_id, metadata in docs_dict.items():
        content = id_to_content.get(doc_id, "")
        all_docs.append({
            "content": content,
            "metadata": metadata
        })
    
    logger.info(f"Préparation de {len(all_docs)} documents pour la construction du graphe")
    
    # Build graph
    graph_manager = GraphManager(chroma_manager, embeddings_manager)
    graph = graph_manager.build_graph(all_docs)
    
    # Save graph for future use
    try:
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"✅ Graphe sauvegardé dans {graph_path}")
    except Exception as e:
        logger.error(f"⚠️ Impossible de sauvegarder le graphe: {str(e)}")
    
    return graph

def expand_results_with_graph(graph, initial_results, n_additional=2):
    """
    Expand search results by traversing the knowledge graph
    
    Args:
        graph: Knowledge graph
        initial_results: Initial results from vector search
        n_additional: Number of additional results to retrieve
        
    Returns:
        List of additional results from graph traversal
    """
    graph_results = []
    processed_ids = set()
    
    # Add initial result IDs to processed set to avoid duplicates
    for result in initial_results:
        if 'id' in result:
            processed_ids.add(result['id'])
    
    # Find nodes that correspond to initial results
    for result in initial_results:
        # Try to find matching node by content or metadata
        result_content = result.get('document', '')
        
        # Find node that most closely matches this result
        matching_node_id = None
        for node_id, node in graph.nodes.items():
            if result_content in node.content:
                matching_node_id = node_id
                break
        
        if not matching_node_id:
            continue
            
        # Find related nodes through different relation types
        for edge in graph.edges:
            if edge.source == matching_node_id:
                target_node_id = edge.target
                
                # Skip if already processed
                if target_node_id in processed_ids:
                    continue
                    
                processed_ids.add(target_node_id)
                
                # Get the target node and create a result
                target_node = graph.nodes[target_node_id]
                
                # Create a result object similar to the ChromaDB results
                graph_result = {
                    'document': target_node.content,
                    'metadata': target_node.metadata,
                    'source_type': 'graph',
                    'relation_type': edge.relation_type,
                    'distance': 1.0 - edge.weight,  # Convert weight to distance
                    'id': target_node_id
                }
                
                graph_results.append(graph_result)
                
                # Limit number of additional results
                if len(graph_results) >= n_additional:
                    return graph_results
    
    return graph_results

def combine_vector_and_graph_results(vector_results, graph_results):
    """
    Merge vector-based and graph-based results, avoiding duplicates
    
    Args:
        vector_results: Results from vector search
        graph_results: Results from graph traversal
        
    Returns:
        List of combined results
    """
    # Start with vector results
    combined_results = list(vector_results)
    
    # Track document contents to avoid duplicates
    seen_contents = set(result.get('document', '')[:100] for result in vector_results)
    
    # Add graph results that aren't duplicates
    for result in graph_results:
        content_preview = result.get('document', '')[:100]
        if content_preview not in seen_contents:
            seen_contents.add(content_preview)
            combined_results.append(result)
    
    return combined_results

def combine_similarity_and_date_hits(similarity_results, date_results, n_context):
    """
    Combine results from similarity search with date-specific results,
    keeping the TOP_K results from similarity search and adding date results as a complement.
    
    Args:
        similarity_results: Results from standard similarity search
        date_results: Results from date-specific search
        n_context: Maximum number of results to return
        
    Returns:
        List of combined results with dates as a complement
    """
    # Start with similarity results
    combined_results = list(similarity_results)
    
    # Track document IDs to avoid duplicates
    seen_ids = set(result.get('id', '') for result in similarity_results)
    
    # Add date results that aren't duplicates
    for result in date_results:
        result_id = result.get('id', '')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            # Mark as date-specific result
            result['from_date_search'] = True
            combined_results.append(result)
    
    # Sort results - prioritizing similarity results first, then date results by their score
    # This ensures that TOP_K similarity results are preserved at the top
    combined_results.sort(key=lambda x: (x.get('from_date_search', False), x.get('distance', 1.0)))
    
    # Limit to requested number of results
    return combined_results[:n_context]

def retrieve_contract_sources_with_alternatives(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False,
                         similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
                         use_alternatives: bool = True, model_name: str = 'mistral-small3.1') -> list:
    """
    Récupère les sources pertinentes en utilisant des requêtes alternatives pour améliorer la recherche

    Args:
        query: Question de l'utilisateur
        n_context: Nombre de chunks pertinents à utiliser
        use_graph: Utiliser le graphe de connaissances
        similarity_threesold: Seuil de similarité pour le filtrage
        use_alternatives: Utiliser des requêtes alternatives
        model_name: Modèle à utiliser pour générer les alternatives

    Returns:
        list: Liste des sources pertinentes avec déduplication
    """
    logger.info(f"Recherche de sources pour: {query}")
    
    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)
    reranker_manager = Reranker("mxbai-rerank-large-v2")

    graph_manager = None
    knowledge_graph = None
    if use_graph:
        logger.info("Utilisation du graphe de connaissances pour enrichir le contexte...")
        graph_manager = GraphManager(chroma_manager, embeddings_manager)
        knowledge_graph = load_or_recreate_knowledge_graph(chroma_manager, embeddings_manager)

    # Structure pour suivre les doublons
    seen_ids = set()
    seen_content_hashes = set()
    all_results = []
    
    # Commencer par la requête originale
    logger.info("Recherche avec la requête originale...")
    original_results = chroma_manager.search(query, n_results=n_context)
    logger.info(f"Trouvé {len(original_results)} résultats pour la requête originale")
    
    # Ajouter les résultats originaux
    for result in original_results:
        result_id = result.get('id', '')
        content = result.get('document', '')
        content_hash = hash(content)
        
        if (result_id and result_id in seen_ids) or (content_hash in seen_content_hashes):
            continue
            
        if result_id:
            seen_ids.add(result_id)
        seen_content_hashes.add(content_hash)
        
        result["source_query"] = "Original"
        all_results.append(result)
    
    # Générer et utiliser des requêtes alternatives si demandé
    if use_alternatives:
        try:
            logger.info("Génération de requêtes alternatives...")
            alternative_queries = generate_queries(query, model_name)
            logger.info(f"Généré {len(alternative_queries)} requêtes alternatives")
            
            # Calculer le nombre de résultats par requête alternative
            remaining_slots = max(0, n_context * 2 - len(all_results))  # Permettre jusqu'à 2x plus de résultats
            results_per_alt_query = max(1, remaining_slots // len(alternative_queries)) if alternative_queries else 0
            
            for i, alt_query in enumerate(alternative_queries, 1):
                logger.info(f"Recherche avec requête alternative {i}: {alt_query}")
                alt_results = chroma_manager.search(alt_query, n_results=results_per_alt_query)
                
                unique_alt_results = 0
                for result in alt_results:
                    result_id = result.get('id', '')
                    content = result.get('document', '')
                    content_hash = hash(content)
                    
                    if (result_id and result_id in seen_ids) or (content_hash in seen_content_hashes):
                        continue
                        
                    if result_id:
                        seen_ids.add(result_id)
                    seen_content_hashes.add(content_hash)
                    
                    result["source_query"] = f"Alternative {i}: {alt_query[:50]}..."
                    all_results.append(result)
                    unique_alt_results += 1
                
                logger.info(f"Ajouté {unique_alt_results} nouveaux résultats uniques de la requête alternative {i}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération des requêtes alternatives: {e}")
            logger.warning("Poursuite avec la requête originale uniquement")
    
    logger.info(f"Total de {len(all_results)} résultats uniques collectés")
    
    # Check if this is a date-related query
    date_related_terms = [
        'date', 'échéance', 'délai', 'terme', 'expire', 'expiration', 'calendrier', 
        'planning', 'horaire', 'jour', 'mois', 'année', 'trimestre', 'semestre',
        'période', 'durée', 'temps', 'chronologie', 'deadline', 'livraison', 
        'anniversaire', 'signature', 'préavis',
        'deadline', 'calendar', 'schedule', 'day', 'month', 'year', 'quarter', 
        'semester', 'period', 'duration', 'time', 'timeline', 'delivery', 
        'anniversary', 'signature', 'notice'
    ]
    is_date_query = any(term in query.lower() for term in date_related_terms)

    # Standard filtering based on similarity threshold
    filtered_results = [d for d in all_results if d['distance'] <= 1-similarity_threesold]
    logger.info(f"Après filtrage par seuil de similarité: {len(filtered_results)} résultats restent")

    # For date-related queries, complement with date-specific results
    if is_date_query:
        logger.info("Requête liée aux dates détectée, sélection de chunks pertinents avec dates")
        date_results = chroma_manager.select_context(all_results, query)
        logger.info(f"Sélection basée sur les dates trouvée: {len(date_results)} résultats avec informations de date")
        
        filtered_results = combine_similarity_and_date_hits(filtered_results, date_results, n_context * 2)
        logger.info(f"Après fusion des résultats de similarité et de date: {len(filtered_results)} résultats totaux")

    # If no results pass the threshold, use the original results
    if not filtered_results and all_results:
        logger.warning("Aucun résultat n'a passé le seuil de similarité. Utilisation des résultats non filtrés à la place.")
        filtered_results = all_results
    
    # Only attempt reranking if we have results
    if filtered_results:
        try:
            # Limiter le nombre de résultats pour le reranking
            rerank_limit = min(len(filtered_results), n_context * 3)  # Rerank jusqu'à 3x le nombre demandé
            reranked_docs = reranker_manager.rerank(query, filtered_results[:rerank_limit], n_context)
            logger.info(f"Reranking réussi de {len(reranked_docs)} documents")
            filtered_results = reranked_docs
        except Exception as e:
            logger.error(f"Erreur lors du reranking: {e}")
            logger.warning("Utilisation des résultats filtrés sans reranking")
            # Limiter aux n_context meilleurs résultats par distance
            filtered_results = sorted(filtered_results, key=lambda x: x.get('distance', 1.0))[:n_context]
    else:
        logger.warning("Aucun document disponible pour le reranking")
    
    # Use graph results if available
    if use_graph and knowledge_graph and filtered_results:
        graph_results = expand_results_with_graph(knowledge_graph, filtered_results, n_additional=2)
        combined_results = combine_vector_and_graph_results(filtered_results, graph_results)
    else:
        combined_results = filtered_results
    
    # Add query info to each result
    for result in combined_results:
        result["query"] = query
    
    return combined_results

def retrieve_contract_sources(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False,
                         similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)),
                         use_hybrid: bool = bool(os.getenv("USE_HYBRID", "True").lower() == "true")) -> list:
    """
    Récupère uniquement les sources pertinentes pour une requête sans générer de réponse LLM

    Args:
        query: Question de l'utilisateur
        n_context: Nombre de chunks pertinents à utiliser
        use_graph: Utiliser le graphe de connaissances
        similarity_threesold: Seuil de similarité pour le filtrage
        use_hybrid: Utiliser la recherche hybride (BM25 + sémantique)

    Returns:
        list: Liste des sources pertinentes
    """
    logger.info(f"Recherche de sources pour: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)
    reranker_manager = Reranker("mxbai-rerank-large-v2")

    graph_manager = None
    knowledge_graph = None
    if use_graph:
        logger.info("Utilisation du graphe de connaissances pour enrichir le contexte...")
        graph_manager = GraphManager(chroma_manager, embeddings_manager)
        # Load or build the graph
        knowledge_graph = load_or_recreate_knowledge_graph(chroma_manager, embeddings_manager)

    # Check if this is a date-related query
    date_related_terms = [
        'date', 'échéance', 'délai', 'terme', 'expire', 'expiration', 'calendrier', 
        'planning', 'horaire', 'jour', 'mois', 'année', 'trimestre', 'semestre',
        'période', 'durée', 'temps', 'chronologie', 'deadline', 'livraison', 
        'anniversaire', 'signature', 'préavis',
        'deadline', 'calendar', 'schedule', 'day', 'month', 'year', 'quarter', 
        'semester', 'period', 'duration', 'time', 'timeline', 'delivery', 
        'anniversary', 'signature', 'notice'
    ]
    is_date_query = any(term in query.lower() for term in date_related_terms)

    # Get search results
    if use_hybrid:
        logger.info("Utilisation de la recherche hybride (vectorielle + BM25)...")
        hybrid_search = get_singleton_hybrid_search(embeddings_manager, chroma_manager)
        # Pour une recherche hybride, le filtrage par seuil est déjà intégré dans la fonction search
        results = hybrid_search.search(query, n_results=n_context, min_semantic_score=similarity_threesold)
        logger.info(f"Found {len(results)} initial hybrid search results")
        
        # Pour les requêtes hybrides, on considère les résultats directement sans filtrage supplémentaire
        filtered_results = results
    else:
        # Standard vector search
        logger.info("Utilisation de la recherche vectorielle standard...")
        results = chroma_manager.search(query, n_results=n_context)
        logger.info(f"Found {len(results)} initial search results")
    
        # Standard filtering based on similarity threshold for non-date queries
        filtered_results = [d for d in results if d['distance'] <= 1-similarity_threesold]
        logger.info(f"After filtering by similarity threshold: {len(filtered_results)} results remain")

    # For date-related queries, complement with date-specific results
    if is_date_query:
        logger.info("Date-related query detected, selecting relevant chunks with dates")
        # Get date-specific results
        date_results = chroma_manager.select_context(results, query)
        logger.info(f"Date-based selection found: {len(date_results)} results with date information")
        
        # Merge similarity results with date-specific results
        filtered_results = combine_similarity_and_date_hits(filtered_results, date_results, n_context * 2)
        logger.info(f"After merging similarity and date results: {len(filtered_results)} total results")

    # If no results pass the threshold, use the original results
    if not filtered_results and results:
        logger.warning("No results passed the similarity threshold. Using unfiltered results instead.")
        filtered_results = results
    
    # Only attempt reranking if we have results
    if filtered_results:
        try:
            reranked_docs = reranker_manager.rerank(query, filtered_results, n_context)
            logger.info(f"Successfully reranked {len(reranked_docs)} documents")
            filtered_results = reranked_docs
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Using filtered results without reranking")
    else:
        logger.warning("No documents available for reranking")
    
    # Use graph results if available
    if use_graph and knowledge_graph:
        graph_results = expand_results_with_graph(knowledge_graph, results, n_additional=2)
        # Combine results (ensuring no duplicates)
        combined_results = combine_vector_and_graph_results(filtered_results, graph_results)
    else:
        combined_results = filtered_results
    
    # Add query info to each result
    for result in combined_results:
        result["query"] = query
    
    return combined_results

def chat_with_contract(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
temperature: float = float(os.getenv("TEMPERATURE", 0.5)), similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)), 
history: str = "", use_hybrid: bool = bool(os.getenv("USE_HYBRID", "True").lower() == "true")) -> tuple:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
        use_hybrid: Whether to use hybrid search (BM25 + semantic)
        
    Returns:
        tuple: (response, sources) where response is the LLM's answer and sources is a list of sources used
    """
    logger.info(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)
    reranker_manager = Reranker("mxbai-rerank-large-v2")

    graph_manager = None
    knowledge_graph = None
    if use_graph:
        logger.info("🔍 Utilisation du graphe de connaissances pour enrichir le contexte...")
        graph_manager = GraphManager(chroma_manager, embeddings_manager)
        # Load or build the graph
        knowledge_graph = load_or_recreate_knowledge_graph(chroma_manager, embeddings_manager)

    # Check if this is a date-related query
    date_related_terms = [
        'date', 'échéance', 'délai', 'terme', 'expire', 'expiration', 'calendrier', 
        'planning', 'horaire', 'jour', 'mois', 'année', 'trimestre', 'semestre',
        'période', 'durée', 'temps', 'chronologie', 'deadline', 'livraison', 
        'anniversaire', 'signature', 'préavis',
        'deadline', 'calendar', 'schedule', 'day', 'month', 'year', 'quarter', 
        'semester', 'period', 'duration', 'time', 'timeline', 'delivery', 
        'anniversary', 'signature', 'notice'
    ]
    is_date_query = any(term in query.lower() for term in date_related_terms)

    # Get search results
    if use_hybrid:
        logger.info("Utilisation de la recherche hybride (vectorielle + BM25)...")
        hybrid_search = get_singleton_hybrid_search(embeddings_manager, chroma_manager)
        # Pour une recherche hybride, le filtrage par seuil est déjà intégré dans la fonction search
        results = hybrid_search.search(query, n_results=n_context, min_semantic_score=similarity_threesold)
        logger.info(f"Found {len(results)} initial hybrid search results")
        
        # Pour les requêtes hybrides, on considère les résultats directement sans filtrage supplémentaire
        filtered_results = results
    else:
        # Standard vector search
        logger.info("Utilisation de la recherche vectorielle standard...")
        results = chroma_manager.search(query, n_results=n_context)
        logger.info(f"Found {len(results)} initial search results")
    
        # Standard filtering based on similarity threshold for non-date queries
        filtered_results = [d for d in results if d['distance'] <= 1-similarity_threesold]
        logger.info(f"After filtering by similarity threshold: {len(filtered_results)} results remain")

    # For date-related queries, complement with date-specific results
    if is_date_query:
        logger.info("Date-related query detected, selecting relevant chunks with dates")
        # Get date-specific results
        date_results = chroma_manager.select_context(results, query)
        logger.info(f"Date-based selection found: {len(date_results)} results with date information")
        
        # Merge similarity results with date-specific results
        filtered_results = combine_similarity_and_date_hits(filtered_results, date_results, n_context * 2)
        logger.info(f"After merging similarity and date results: {len(filtered_results)} total results")

    # If no results pass the threshold, use the original results
    if not filtered_results and results:
        logger.warning("No results passed the similarity threshold. Using unfiltered results instead.")
        filtered_results = results
    
    # Only attempt reranking if we have results
    if filtered_results:
        try:
            reranked_docs = reranker_manager.rerank(query, filtered_results, n_context)
            logger.info(f"Successfully reranked {len(reranked_docs)} documents")
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Using filtered results without reranking")
            reranked_docs = filtered_results
    else:
        logger.warning("No documents available for reranking")
        reranked_docs = []
    
    # Use graph results if available
    if use_graph and knowledge_graph:
        graph_results = expand_results_with_graph(knowledge_graph, results, n_additional=2)
        # Combine results (ensuring no duplicates)
        combined_results = combine_vector_and_graph_results(filtered_results, graph_results)
    else:
        combined_results = filtered_results
    
    # Check if we have any results to use
    if not combined_results:
        no_results_response = "Je n'ai pas trouvé d'information pertinente dans la base de connaissances pour répondre à votre question. Pourriez-vous reformuler ou préciser votre demande?"
        print("\n🤖 Réponse :")
        print(no_results_response)
        logger.warning("No relevant documents found for the query")
        return no_results_response, []
    
    # Prepare context for the prompt
    context_parts = []
    for result in combined_results:
        # En-tête avec les métadonnées
        header = (
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Hiérarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}"
        )

        # Ajouter les dates si présentes dans les métadonnées
        if result['metadata'].get('dates'):
            header += f"\nDates: {result['metadata'].get('dates')}"

        # Contenu adapté selon le type (résumé ou original)
        if result.get("is_summary", False):
            content = (
                f"\nRésumé:\n{result['document']}\n"
                f"Contenu détaillé si nécessaire:\n{result.get('original_content', 'Non disponible')}"
            )
        else:
            content = f"\nContenu:\n{result['document']}"

        # Ajouter la source au contexte
        context_parts.append(f"{header}\n{content}")

    # Joindre toutes les parties du contexte
    context = "\n\n---\n\n".join(context_parts)

    # Create the prompt with context
    prompt = f"""
    You are an assistant specializing in contract analysis.

    Here is the history of the conversation:
    {history}

    Here is the relevant context extracted from the documents. …
    {context}

    User's question: {query}

    Always double-check any summary by consulting the detailed content.
    If the context lacks the information, state it explicitly.
    If a history is present and non-null, extract useful elements from it to answer the initial question.
    """


    # Get response from Ollama
    response = llm_chat_call_with_ollama(prompt, temperature, model, context_window)
    logger.info("\n🤖 Réponse :")
    logger.info(response)
    print("\n🤖 Réponse :")
    print(response)
    print("\n📚 Sources :")
    print("=" * 80)

    # Display sources with metadata
    logger.info("\n📚 Sources :")
    logger.info("=" * 80)
    for i, result in enumerate(combined_results, 1):
        logger.info("\n" + "-" * 40)
        logger.info(f"\nSource {i}/{len(combined_results)}")
        
        # Afficher le type de source (résumé ou original)
        if result.get("source_type") == "graph":
            logger.info("📊 Source obtenue via le graphe de connaissances")
            logger.info(f"Relation: {result.get('relation_type', 'Non spécifié')}")
        logger.info("-" * 40)
        logger.info(f"Hierarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}")
        logger.info(f"Document: {result['metadata'].get('document_title', 'Non spécifié')}")

        # Afficher les dates si présentes
        if result['metadata'].get('dates'):
            logger.info(f"Dates: {result['metadata'].get('dates')}")
            print(f"Dates: {result['metadata'].get('dates')}")

        logger.info(f"Distance: {result['distance']:.4f}")
        
        # Afficher les scores BM25 si disponibles
        if 'bm25_score' in result:
            logger.info(f"Score BM25: {result['bm25_score']:.4f}")
            logger.info(f"Score combiné: {result['combined_score']:.4f}")
            print(f"Score BM25: {result['bm25_score']:.4f}")
            print(f"Score combiné: {result['combined_score']:.4f}")

        # Afficher le contenu
        if result.get("is_summary", False):
            logger.info("\nRésumé utilisé:")
            logger.info(result["document"])
            logger.info("\nContenu original:")
            logger.info(result.get("original_content", "Non disponible") + "...")
        else:
            logger.info("\nContenu:")
            logger.info(result["document"] + "...")

        logger.info("-" * 40)

    # Afficher les statistiques
    summaries = sum(1 for r in combined_results if r.get("is_summary", False))
    graph_sources = sum(1 for r in combined_results if r.get("source_type") == "graph")
    date_sources = sum(1 for r in combined_results if r.get('metadata', {}).get('dates'))
    hybrid_sources = sum(1 for r in combined_results if r.get('hybrid_search', False))
    
    logger.info(f"\n📊 Statistiques des sources:")
    logger.info(f"- Total: {len(combined_results)}")
    logger.info(f"- Résumés: {summaries}")
    logger.info(f"- Contenus originaux: {len(combined_results) - summaries - graph_sources}")
    logger.info(f"- Sources du graphe: {graph_sources}")
    logger.info(f"- Sources avec dates: {date_sources}")
    logger.info(f"- Sources via recherche hybride: {hybrid_sources}")
    
    return response

def chat_with_contract_using_query_decomposition(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
temperature: float = float(os.getenv("TEMPERATURE", 0.5)), similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)),
max_total_sources: int = 15) -> tuple:
    """
    Chat with the contract by decomposing the query into sub-queries, getting answers for each,
    and then synthesizing a final response with all sources.

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
        temperature: Temperature for LLM generation
        similarity_threesold: Threshold for similarity filtering
        model: LLM model to use
        context_window: Context window size for the LLM
        max_total_sources: Maximum total number of sources to collect across all sub-queries
    """
    logger.info(f"\n💬 Requête complexe: {query}")
    
    # Structure pour suivre les doublons
    seen_ids = set()
    seen_content_hashes = set()
    all_sources = []
    
    # Commencer par récupérer les sources pour la question principale
    logger.info("Recherche de sources pour la question principale...")
    
    # Allocation plus équilibrée: environ 1/3 pour la question principale au lieu de 1/2
    # avec un minimum de 2 sources et un maximum de n_context/3
    main_question_sources = max(2, min(n_context // 3, 3))
    
    main_sources = retrieve_contract_sources(query, main_question_sources, use_graph, similarity_threesold)
    logger.info(f"Trouvé {len(main_sources)} sources pour la question principale")
    
    # Filtrer et ajouter les sources principales
    main_unique_sources = []
    for source in main_sources:
        source_id = source.get('id', '')
        content = source.get('document', '')
        content_hash = hash(content)
        
        # Vérifier si cette source est un doublon
        if (source_id and source_id in seen_ids) or (content_hash in seen_content_hashes):
            continue
            
        # Marquer comme vue
        if source_id:
            seen_ids.add(source_id)
        seen_content_hashes.add(content_hash)
        
        # Ajouter la question principale à la source
        source["sub_query"] = "Question principale"
        main_unique_sources.append(source)
        all_sources.append(source)
    
    logger.info(f"Après déduplication: {len(main_unique_sources)}/{len(main_sources)} sources uniques pour la question principale")
    
    # Ensuite, décomposer la requête en sous-questions
    sub_queries = define_subqueries(query)
    logger.info(f"Requête décomposée en {len(sub_queries.questions)} sous-questions")
    
    # Ajuster dynamiquement le nombre de sources par sous-requête
    # Répartir les sources restantes entre les sous-questions
    remaining_sources = max_total_sources - len(all_sources)
    num_subqueries = len(sub_queries.questions)
    
    if num_subqueries > 0 and remaining_sources > 0:
        # Garantir au moins 2 sources par sous-question si possible
        sources_per_subquery = max(2, remaining_sources // num_subqueries)
        
        # Si le nombre calculé dépasse n_context, on le plafonne
        # Cela permet d'éviter de demander trop de sources pour une seule sous-question
        sources_per_subquery = min(sources_per_subquery, n_context // 2)
        
        logger.info(f"Sources par sous-requête: {sources_per_subquery} (total restant: {remaining_sources})")
    else:
        sources_per_subquery = 0
        logger.info("Aucune source supplémentaire ne sera collectée pour les sous-requêtes")
    
    sub_questions_with_sources = []
    
    # Ajouter la question principale avec ses sources
    sub_questions_with_sources.append({
        "question": query,
        "sources": main_unique_sources,
        "is_main_question": True
    })
    
    # Traiter les sous-questions seulement si on a de la place pour plus de sources
    if sources_per_subquery > 0:
        for i, sub_query in enumerate(sub_queries.questions, 1):
            logger.info(f"\n🔍 Sous-question {i}/{num_subqueries}: {sub_query}")
            
            # Récupérer uniquement les sources sans générer de réponse
            sources = retrieve_contract_sources(sub_query, sources_per_subquery, use_graph, similarity_threesold)
            logger.info(f"Trouvé {len(sources)} sources pour la sous-question")
            
            # Filtrer les sources pour éviter les doublons
            unique_sources_for_query = []
            for source in sources:
                source_id = source.get('id', '')
                content = source.get('document', '')
                content_hash = hash(content)
                
                # Vérifier si cette source est un doublon
                if (source_id and source_id in seen_ids) or (content_hash in seen_content_hashes):
                    logger.debug(f"Source dupliquée ignorée: {source_id}")
                    continue
                    
                # Marquer comme vue
                if source_id:
                    seen_ids.add(source_id)
                seen_content_hashes.add(content_hash)
                
                # Ajouter la sous-question à la source
                source["sub_query"] = sub_query
                unique_sources_for_query.append(source)
                all_sources.append(source)
            
            logger.info(f"Après déduplication: {len(unique_sources_for_query)}/{len(sources)} sources uniques pour cette sous-question")
            
            # Stocker la sous-question avec ses sources uniques
            sub_questions_with_sources.append({
                "question": sub_query, 
                "sources": unique_sources_for_query,
                "is_main_question": False
            })
            
            # Si on atteint le nombre maximum de sources, arrêter
            if len(all_sources) >= max_total_sources:
                logger.info(f"Nombre maximum de sources atteint ({max_total_sources})")
                break
    
    logger.info(f"Total de {len(all_sources)} sources uniques collectées pour la question principale et les sous-questions")
    
    # Si aucune source n'a été trouvée
    if not all_sources:
        no_results_response = "Je n'ai pas trouvé d'information pertinente dans la base de connaissances pour répondre à votre question ou ses sous-questions. Pourriez-vous reformuler ou préciser votre demande?"
        print("\n🤖 Réponse :")
        print(no_results_response)
        logger.warning("Aucune source pertinente trouvée pour les sous-questions")
        return no_results_response, []
    
    # Préparer le contexte pour la synthèse finale
    context_parts = []
    for source in all_sources:
        # En-tête avec les métadonnées
        header = (
            f"Document: {source['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {source['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Hiérarchie: {source['metadata'].get('hierarchy', 'Non spécifié')}\n"
            f"En réponse à: {source.get('sub_query', 'Question principale')}"
        )

        # Ajouter les dates si présentes dans les métadonnées
        if source['metadata'].get('dates'):
            header += f"\nDates: {source['metadata'].get('dates')}"

        # Contenu adapté selon le type
        if source.get("is_summary", False):
            content = (
                f"\nRésumé:\n{source['document']}\n"
                f"Contenu détaillé si nécessaire:\n{source.get('original_content', 'Non disponible')}"
            )
        else:
            content = f"\nContenu:\n{source['document']}"

        # Ajouter la source au contexte
        context_parts.append(f"{header}\n{content}")

    # Joindre toutes les parties du contexte
    context = "\n\n---\n\n".join(context_parts)
    
    # Estimer la taille du contexte en tokens
    # Une estimation courante est d'environ 4 caractères par token en moyenne pour les langues européennes
    context_token_estimate = len(context) / 4
    
    # Réserve pour le prompt (instructions, question, etc.) - environ 1000 tokens
    prompt_overhead = 1000
    available_context_tokens = context_window - prompt_overhead
    
    # Vérifier si le contexte est trop grand et l'ajuster si nécessaire
    if context_token_estimate > available_context_tokens:
        logger.warning(f"Contexte trop grand: ~{int(context_token_estimate)} tokens estimés > {available_context_tokens} disponibles")
        
        # Stratégie d'ajustement: réduire proportionnellement le contenu de chaque source
        reduction_ratio = available_context_tokens / context_token_estimate
        logger.info(f"Réduction du contexte à {int(reduction_ratio * 100)}% de sa taille originale")
        
        # Reconstruire le contexte avec contenu réduit
        adjusted_context_parts = []
        for source in all_sources:
            # Conserver l'en-tête complet
            header = (
                f"Document: {source['metadata'].get('document_title', 'Non spécifié')}\n"
                f"Section: {source['metadata'].get('section_number', 'Non spécifié')}\n"
                f"Hiérarchie: {source['metadata'].get('hierarchy', 'Non spécifié')}\n"
                f"En réponse à: {source.get('sub_query', 'Question principale')}"
            )
            
            if source['metadata'].get('dates'):
                header += f"\nDates: {source['metadata'].get('dates')}"
                
            # Réduire proportionnellement le contenu
            content = source.get('document', '')
            max_content_length = int(len(content) * reduction_ratio)
            if len(content) > max_content_length:
                content = content[:max_content_length] + " [...]"
                
            adjusted_part = f"{header}\n\nContenu:\n{content}"
            adjusted_context_parts.append(adjusted_part)
            
        # Reconstruire le contexte
        context = "\n\n---\n\n".join(adjusted_context_parts)
        new_token_estimate = len(context) / 4
        logger.info(f"Taille de contexte ajustée: ~{int(new_token_estimate)} tokens estimés")
    
    # Extraire la liste des sous-questions (sans la question principale)
    sub_questions_list = [q["question"] for q in sub_questions_with_sources if not q.get("is_main_question", False)]
    
    # Créer le prompt pour la synthèse finale
    final_prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats.

J'ai décomposé la question de l'utilisateur en plusieurs sous-questions:
{", ".join(sub_questions_list)}

Voici les extraits pertinents de documents pour répondre à la question principale et aux sous-questions:

{context}

Question principale de l'utilisateur: {query}

En utilisant uniquement les informations fournies dans ces extraits de documents, synthétise une réponse complète et cohérente 
à la question principale. Organise ta réponse de manière logique et structurée. Si certaines sous-questions ne peuvent pas être 
répondues avec les informations disponibles, indique-le clairement.
"""
    
    # Obtenir la synthèse finale
    final_response = llm_chat_call_with_ollama(final_prompt, temperature, model, context_window)
    
    # Afficher la réponse finale
    logger.info("\n🤖 Réponse synthétisée:")
    logger.info(final_response)
    print("\n🤖 Réponse synthétisée:")
    print(final_response)
    
    # Afficher toutes les sources utilisées
    print(f"\n📚 Sources utilisées ({len(all_sources)} au total):")
    print("=" * 80)
    
    # Afficher les sources regroupées par sous-question
    sources_by_query = {}
    for source in all_sources:
        sub_query = source.get("sub_query", "Question principale")
        if sub_query not in sources_by_query:
            sources_by_query[sub_query] = []
        sources_by_query[sub_query].append(source)
    
    # Afficher d'abord les sources pour la question principale
    if "Question principale" in sources_by_query:
        main_sources = sources_by_query["Question principale"]
        print(f"\n--- Sources pour la question principale ({len(main_sources)} sources) ---")
        
        for i, source in enumerate(main_sources, 1):
            print(f"\nSource {i}/{len(main_sources)}")
            
            # Afficher les informations de la source
            print(f"Document: {source['metadata'].get('document_title', 'Non spécifié')}")
            print(f"Hiérarchie: {source['metadata'].get('hierarchy', 'Non spécifié')}")
            
            # Afficher les dates si présentes
            if source['metadata'].get('dates'):
                print(f"Dates: {source['metadata'].get('dates')}")
                
            # Afficher un extrait du contenu
            content_preview = source['document'][:200] + "..." if len(source['document']) > 200 else source['document']
            print(f"Extrait: {content_preview}")
            print("-" * 40)
    
    # Puis les sources pour chaque sous-question
    for sub_query, sources in sources_by_query.items():
        if sub_query == "Question principale":
            continue  # Déjà affiché
            
        print(f"\n--- Sources pour: '{sub_query}' ({len(sources)} sources) ---")
        
        for i, source in enumerate(sources, 1):
            print(f"\nSource {i}/{len(sources)}")
            
            # Afficher les informations de la source
            print(f"Document: {source['metadata'].get('document_title', 'Non spécifié')}")
            print(f"Hiérarchie: {source['metadata'].get('hierarchy', 'Non spécifié')}")
            
            # Afficher les dates si présentes
            if source['metadata'].get('dates'):
                print(f"Dates: {source['metadata'].get('dates')}")
                
            # Afficher un extrait du contenu
            content_preview = source['document'][:200] + "..." if len(source['document']) > 200 else source['document']
            print(f"Extrait: {content_preview}")
            print("-" * 40)
    
    return final_response, all_sources

def chat_with_contract_using_query_alternatives(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
temperature: float = float(os.getenv("TEMPERATURE", 0.5)), similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)),
use_alternatives: bool = True) -> tuple:
    """
    Chat with the contract using alternative queries to improve retrieval

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
        temperature: Temperature for LLM generation
        similarity_threesold: Threshold for similarity filtering
        model: LLM model to use
        context_window: Context window size for the LLM
        use_alternatives: Whether to use alternative queries for better retrieval
        
    Returns:
        tuple: (response, sources) where response is the LLM's answer and sources is a list of sources used
    """
    logger.info(f"\n💬 Chat avec requêtes alternatives: {query}")

    # Get sources using alternative queries
    combined_results = retrieve_contract_sources_with_alternatives(
        query, n_context, use_graph, similarity_threesold, use_alternatives, model.split(':')[0]
    )
    
    # Check if we have any results to use
    if not combined_results:
        no_results_response = "Je n'ai pas trouvé d'information pertinente dans la base de connaissances pour répondre à votre question. Pourriez-vous reformuler ou préciser votre demande?"
        print("\n🤖 Réponse :")
        print(no_results_response)
        logger.warning("No relevant documents found for the query")
        return no_results_response, []
    
    # Prepare context for the prompt
    context_parts = []
    for result in combined_results:
        # En-tête avec les métadonnées
        header = (
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Hiérarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}\n"
            f"Source de recherche: {result.get('source_query', 'Original')}"
        )

        # Ajouter les dates si présentes dans les métadonnées
        if result['metadata'].get('dates'):
            header += f"\nDates: {result['metadata'].get('dates')}"

        # Contenu adapté selon le type (résumé ou original)
        if result.get("is_summary", False):
            content = (
                f"\nRésumé:\n{result['document']}\n"
                f"Contenu détaillé si nécessaire:\n{result.get('original_content', 'Non disponible')}"
            )
        else:
            content = f"\nContenu:\n{result['document']}"

        # Ajouter la source au contexte
        context_parts.append(f"{header}\n{content}")

    # Joindre toutes les parties du contexte
    context = "\n\n---\n\n".join(context_parts)

    # Create the prompt with context
    prompt = f"""You are an assistant specializing in contract analysis.
    Here is the relevant context extracted from the documents using multiple search strategies (original query + alternative formulations).
    For each section, you either have a summary with the detailed content available or the original content itself.
    First use the summaries to get an overview, then consult the detailed content if you need more precision.

    {context}

    User's question: {query}

    Provide a precise answer based on the context given.
    If you use a summary, check the detailed content to ensure your answer's accuracy.
    If you can't find the information in the context, state that clearly."""

    # Get response from Ollama
    response = llm_chat_call_with_ollama(prompt, temperature, model, context_window)
    logger.info("\n🤖 Réponse :")
    logger.info(response)
    print("\n🤖 Réponse :")
    print(response)
    print("\n📚 Sources (avec requêtes alternatives) :")
    print("=" * 80)

    # Display sources with metadata
    logger.info("\n📚 Sources (avec requêtes alternatives) :")
    logger.info("=" * 80)
    
    # Group sources by search query
    sources_by_query = {}
    for result in combined_results:
        source_query = result.get('source_query', 'Original')
        if source_query not in sources_by_query:
            sources_by_query[source_query] = []
        sources_by_query[source_query].append(result)
    
    # Display sources grouped by query type
    for query_type, sources in sources_by_query.items():
        logger.info(f"\n--- Sources de: {query_type} ({len(sources)} sources) ---")
        print(f"\n--- Sources de: {query_type} ({len(sources)} sources) ---")
        
        for i, result in enumerate(sources, 1):
            logger.info("\n" + "-" * 40)
            logger.info(f"\nSource {i}/{len(sources)}")
            
            # Afficher le type de source (résumé ou original)
            if result.get("source_type") == "graph":
                logger.info("📊 Source obtenue via le graphe de connaissances")
                logger.info(f"Relation: {result.get('relation_type', 'Non spécifié')}")
                print("📊 Source obtenue via le graphe de connaissances")
                print(f"Relation: {result.get('relation_type', 'Non spécifié')}")
            
            logger.info("-" * 40)
            logger.info(f"Hierarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}")
            logger.info(f"Document: {result['metadata'].get('document_title', 'Non spécifié')}")
            print(f"Hierarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}")
            print(f"Document: {result['metadata'].get('document_title', 'Non spécifié')}")

            # Afficher les dates si présentes
            if result['metadata'].get('dates'):
                logger.info(f"Dates: {result['metadata'].get('dates')}")
                print(f"Dates: {result['metadata'].get('dates')}")

            logger.info(f"Distance: {result['distance']:.4f}")
            print(f"Distance: {result['distance']:.4f}")

            # Afficher le contenu
            if result.get("is_summary", False):
                logger.info("\nRésumé utilisé:")
                logger.info(result["document"][:200] + "...")
                print(f"Contenu: {result['document'][:200]}...")
            else:
                logger.info("\nContenu:")
                logger.info(result["document"][:200] + "...")
                print(f"Contenu: {result['document'][:200]}...")

            logger.info("-" * 40)

    # Afficher les statistiques
    original_sources = len(sources_by_query.get('Original', []))
    alternative_sources = sum(len(sources) for query_type, sources in sources_by_query.items() if query_type != 'Original')
    graph_sources = sum(1 for r in combined_results if r.get("source_type") == "graph")
    date_sources = sum(1 for r in combined_results if r.get('metadata', {}).get('dates'))
    hybrid_sources = sum(1 for r in combined_results if r.get('hybrid_search', False))
    
    logger.info(f"\n📊 Statistiques des sources:")
    logger.info(f"- Total: {len(combined_results)}")
    logger.info(f"- Sources requête originale: {original_sources}")
    logger.info(f"- Sources requêtes alternatives: {alternative_sources}")
    logger.info(f"- Sources du graphe: {graph_sources}")
    logger.info(f"- Sources avec dates: {date_sources}")
    logger.info(f"- Sources via recherche hybride: {hybrid_sources}")
    
    print(f"\n📊 Statistiques des sources:")
    print(f"- Total: {len(combined_results)}")
    print(f"- Sources requête originale: {original_sources}")
    print(f"- Sources requêtes alternatives: {alternative_sources}")
    print(f"- Sources du graphe: {graph_sources}")
    print(f"- Sources avec dates: {date_sources}")
    print(f"- Sources via recherche hybride: {hybrid_sources}")
    
    return response, combined_results

def query_classifier(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
                 temperature: float = float(os.getenv("TEMPERATURE", 0.5)), 
                 similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
                 model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), 
                 context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)),
                 use_classification: bool = False,
                 use_hybrid: bool = bool(os.getenv("USE_HYBRID", "True").lower() == "true")) -> tuple:
    """
    Process a user query by first classifying it as RAG or LLM and then routing to the appropriate handler.
    
    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
        temperature: Temperature for LLM generation
        similarity_threesold: Threshold for similarity filtering
        model: LLM model to use
        context_window: Context window size for the LLM
        use_classification: Whether to use query classification (if False, always use RAG)
        use_hybrid: Whether to use hybrid search (BM25 + semantic)
        
    Returns:
        tuple: (response, sources) where response is the LLM's answer and sources is a list of sources used
    """

    similar_conversations = retrieve_similar_conversations(query, os.getenv("HISTORY_DB_FILE"))

    history_summary = ""

    if similar_conversations:
            history_summary = "\n".join(
                [
                    f"User: {conv['question']}\nAssistant: {conv['response']}"
                    for conv in similar_conversations
                ]
            )

            if "#force" not in query:

                prompt = f"""
                    You are an assistant specializing in contract analysis.

                    Here is the history of the conversation:
                    {history_summary}

                    User's question: {query}

                    Always double-check any summary by consulting the detailed content.
                    If the context lacks the information, state it explicitly.
                    If a history is present and non-null, extract useful elements from it to answer the initial question.
                    """
                

                response = llm_chat_call_with_ollama(prompt, temperature, model, context_window)
                add_conversation_in_history_db(
                    os.getenv("HISTORY_DB_FILE"), query, response
                )
                print("\n🤖 Réponse :")
                print(response)
                return response

    if "#force" in query:
        query = query.replace("#force", "")

    if history_summary == "":
        full_history = get_history(os.getenv("HISTORY_DB_FILE"))
        # on prend les 8 dernières entrées (4 paires)
        recent = full_history[-8:]
        # regrouper en paires User/Assistant
        pairs = []
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                user_msg = recent[i]["content"]
                assistant_msg = recent[i + 1]["content"]
                pairs.append((user_msg, assistant_msg))
        # on garde au plus les 4 dernières paires
        pairs = pairs[-4:]
        # transformer en résumé textuel
        history_summary = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in pairs]
        )
        logger.debug(
            f"🔍 Résumé fallback de l'historique (4 derniers échanges) :\n{history_summary}"
        )
    else:
        # on garde au plus 4 messages similaires trouvés
        similar_messages = similar_messages[:4]
        history_summary = "\n".join(
            [
                f"User: {conv['question']}\nAssistant: {conv['response']}"
                for conv in similar_messages
            ]
        )
        logger.debug(
            f"🔍 Résumé historique (similar messages) :\n{history_summary}"
        )

    if use_classification:
        # Classify the query as either "RAG" or "LLM"
        classification = determine_inference_mode(query, verbose=True)
        logger.info(f"Query classification result: {classification}")
        
        if classification == "llm":
            # For LLM queries, use direct generation without context
            logger.info(f"Using direct LLM processing for query: {query}")
            # Create a simple prompt for the LLM
            prompt = f"""You are a contract analysis assistant. 
            Answer the following question about contracts to the best of your ability, based on general knowledge.
            If you're uncertain, mention that you don't have specific contract details.
            
            Question: {query}
            """
            response = llm_chat_call_with_ollama(prompt=prompt, temperature=temperature, model=model, context_window=context_window)
            logger.info("\n🤖 Réponse :")
            logger.info(response)
            print("\n🤖 Réponse :")
            print(response)
            add_conversation_in_history_db(
                os.getenv("HISTORY_DB_FILE"), query, response
            )
            return response  # No sources for LLM-only processing
        
    # If classification is "RAG" or classification is disabled, use RAG processing
    logger.info(f"Using RAG processing for query: {query}")
    response = chat_with_contract(
        query, n_context, use_graph, temperature, 
        similarity_threesold, model, context_window, 
        history_summary, use_hybrid
    )
    add_conversation_in_history_db(
            os.getenv("HISTORY_DB_FILE"), query, response
    )
