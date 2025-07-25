from document_processing.llm_chat import ask_ollama
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from core.graph_manager import GraphManager
from document_processing.reranker import Reranker
from utils.logger import setup_logger
import os
from dotenv import load_dotenv
# Import the query classification
from core.query_classification_v2 import route_and_execute
from core.history_func import retrieve_similar_conversations, add_conversation_with_embedding, get_history

# Configurer le logger pour ce module
logger = setup_logger(__file__)
load_dotenv("config.env")

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

def merge_date_results(similarity_results, date_results, n_context):
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

def chat_with_contract(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
temperature: float = float(os.getenv("TEMPERATURE", 0.5)), similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)), 
history: str = "") -> tuple:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
        
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
        knowledge_graph = load_or_build_graph(chroma_manager, embeddings_manager)

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
        filtered_results = merge_date_results(filtered_results, date_results, n_context * 2)
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
        graph_results = get_graph_augmented_results(knowledge_graph, results, n_additional=2)
        # Combine results (ensuring no duplicates)
        combined_results = merge_results(filtered_results, graph_results)
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
        ### SYSTEM
        You are a senior legal-tech assistant specialized in contract analysis.

        ### CONTEXT
        Conversation history:
        {history}

        Extracts from source documents:
        {context}

        ### PERMANENT RULES
        1. Work in French.
        2. Quote exactly every article/clause/date/amount you reuse and enclose it in « » quotation marks.
        3. If the required info is absent, explicitly say: “Information non trouvée dans le contexte fourni.”
        4. Answer in well-structured markdown (##, ###).
        5. Never invent numbers or dates.

        ### CONDITIONAL RULES
        Recherchez attentivement toutes les valeurs numériques, dates et
        spécifications techniques dans le contexte fourni.
        Citez-les exactement et vérifiez chaque occurrence avant de répondre.
        Relisez attentivement le contexte et double-vérifiez tout résumé.


        ### USER QUESTION
        {query}

        ### OUTPUT FORMAT
        **Réponse :** (en markdown)
    """

    # Get response from Ollama
    response = ask_ollama(prompt, temperature, model, context_window)
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
    
    logger.info(f"\n📊 Statistiques des sources:")
    logger.info(f"- Total: {len(combined_results)}")
    logger.info(f"- Résumés: {summaries}")
    logger.info(f"- Contenus originaux: {len(combined_results) - summaries - graph_sources}")
    logger.info(f"- Sources du graphe: {graph_sources}")
    logger.info(f"- Sources avec dates: {date_sources}")
    
    return response

def process_query(query: str, n_context: int = int(os.getenv("TOP_K", 5)), use_graph: bool = False, 
                 temperature: float = float(os.getenv("TEMPERATURE", 0.5)), 
                 similarity_threesold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.6)), 
                 model: str = os.getenv("LLM_MODEL", "mistral-small3.1:latest"), 
                 context_window: int = int(os.getenv("CONTEXT_WINDOW", 0)),
                 use_classification: bool = False) -> tuple:
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
                    ### SYSTEM
                    You are a senior legal-tech assistant specialized in contract analysis.

                    ### CONTEXT
                    Conversation history:
                    {history_summary}

                    ### PERMANENT RULES
                    1. Work in French.
                    2. Quote exactly every article/clause/date/amount you reuse and enclose it in « » quotation marks.
                    3. If the required info is absent, explicitly say: “Information non trouvée dans le contexte fourni.”
                    4. Answer in well-structured markdown (##, ###).
                    5. Never invent numbers or dates.

                    ### CONDITIONAL RULES
                    Recherchez attentivement toutes les valeurs numériques, dates et
                    spécifications techniques dans le contexte fourni.
                    Citez-les exactement et vérifiez chaque occurrence avant de répondre.
                    Relisez attentivement le contexte et double-vérifiez tout résumé.


                    ### USER QUESTION
                    {query}

                    ### OUTPUT FORMAT
                    **Réponse :** (en markdown)
                """
                

                response = ask_ollama(prompt, temperature, model, context_window)
                add_conversation_with_embedding(
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
        classification = route_and_execute(query, verbose=True)
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
            response = ask_ollama(prompt=prompt, temperature=temperature, model=model, context_window=context_window)
            logger.info("\n🤖 Réponse :")
            logger.info(response)
            print("\n🤖 Réponse :")
            print(response)
            add_conversation_with_embedding(
                os.getenv("HISTORY_DB_FILE"), query, response
            )
            return response  # No sources for LLM-only processing
        
    # If classification is "RAG" or classification is disabled, use RAG processing
    logger.info(f"Using RAG processing for query: {query}")

    response = chat_with_contract(
        query, n_context, use_graph, temperature, 
        similarity_threesold, model, context_window, 
        history_summary
    )
    add_conversation_with_embedding(
            os.getenv("HISTORY_DB_FILE"), query, response
    )
