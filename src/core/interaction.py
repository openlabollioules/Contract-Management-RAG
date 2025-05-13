from document_processing.llm_chat import ask_ollama
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from core.graph_manager import GraphManager
from document_processing.reranker import Reranker
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def search_contracts(query: str, n_results: int = 5) -> None:
    """
    Search in the contract database

    Args:
        query: Search query
        n_results: Number of results to return
    """
    logger.info(f"\n🔍 Recherche: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)

    # Search
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
        logger.info(f"Distance: {result['distance']:.4f}")

def load_or_build_graph(chroma_manager, embeddings_manager):
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

def get_graph_augmented_results(graph, initial_results, n_additional=2):
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

def merge_results(vector_results, graph_results):
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

def chat_with_contract(query: str, n_context: int = 5, use_graph: bool = False, temperature: float = 0.5, similarity_threesold: float = 0.6, model: str = "mistral-small3.1:latest") -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
    """
    logger.info(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)
    reranker_manager = Reranker("bge-reranker-large")

    graph_manager = None
    knowledge_graph = None
    if use_graph:
        logger.info("🔍 Utilisation du graphe de connaissances pour enrichir le contexte...")
        graph_manager = GraphManager(chroma_manager, embeddings_manager)
        # Load or build the graph
        knowledge_graph = load_or_build_graph(chroma_manager, embeddings_manager)

    results = chroma_manager.search(query, n_results=n_context)
    filtered_results = [d for d in results if d['distance'] <= 1-similarity_threesold]
    reranked_docs = reranker_manager.rerank(query, filtered_results, n_context)
    print(f"voila les résultats : {filtered_results}")
    
    if use_graph and knowledge_graph:
        graph_results = get_graph_augmented_results(knowledge_graph, results, n_additional=2)
        # Combine results (ensuring no duplicates)
        combined_results = merge_results(filtered_results, graph_results)
    else:
        combined_results = filtered_results
    
    # Prepare context for the prompt
    context_parts = []
    for result in combined_results:
        # En-tête avec les métadonnées
        header = (
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Hiérarchie: {result['metadata'].get('hierarchy', 'Non spécifié')}"
        )

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
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats. 
Voici le contexte pertinent extrait des documents. Pour chaque section, tu as soit un résumé avec le contenu détaillé disponible, soit directement le contenu original.
Utilise d'abord les résumés pour avoir une vue d'ensemble, puis consulte les contenus détaillés si nécessaire pour plus de précision.

{context}

Question de l'utilisateur : {query}

Réponds de manière précise en te basant sur le contexte fourni.
Si tu utilises un résumé, vérifie dans le contenu détaillé pour t'assurer de la précision de ta réponse.
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""

    # Get response from Ollama
    response = ask_ollama(prompt, temperature)
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
        logger.info(f"Hierarchie: {result["metadata"].get("hierarchy")}")
        logger.info(f"Document: {result["metadata"].get("document_title")}")

        logger.info(f"Distance: {result['distance']:.4f}")

        # Afficher le contenu
        if result.get("is_summary", False):
            logger.info("\nRésumé utilisé:")
            logger.info(result["document"])
            logger.info("\nContenu original:")
            logger.info(result.get("original_content", "Non disponible")[:200] + "...")
        else:
            logger.info("\nContenu:")
            logger.info(result["document"][:200] + "...")

        logger.info("-" * 40)

    # Afficher les statistiques
    summaries = sum(1 for r in combined_results if r.get("is_summary", False))
    graph_sources = sum(1 for r in combined_results if r.get("source_type") == "graph")
    logger.info(f"\n📊 Statistiques des sources:")
    logger.info(f"- Total: {len(combined_results)}")
    logger.info(f"- Résumés: {summaries}")
    logger.info(f"- Contenus originaux: {len(combined_results) - summaries - graph_sources}")
    logger.info(f"- Sources du graphe: {graph_sources}")
    return response
