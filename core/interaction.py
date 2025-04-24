from document_processing.llm_chat import ask_ollama
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
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


def chat_with_contract(query: str, n_context: int = 3) -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
    """
    logger.info(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)

    # Search for relevant context
    results = chroma_manager.search(query, n_results=n_context)

    # Prepare context for the prompt
    context = "\n\n".join(
        [
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Chapter: {result['metadata'].get('chapter_title', 'Non spécifié')}\n"
            f"Content: {result['document']}"
            for result in results
        ]
    )

    # Create the prompt with context
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats. 
Voici le contexte pertinent extrait des documents :

{context}

Question de l'utilisateur : {query}

Réponds de manière précise en te basant uniquement sur le contexte fourni. 
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""

    # Get response from Ollama
    response = ask_ollama(prompt)
    logger.info("\n🤖 Réponse :")
    logger.info(response)

    # Display sources with metadata
    logger.info("\n📚 Sources :")
    logger.info("=" * 80)
    for i, result in enumerate(results, 1):
        logger.info("\n" + "-" * 40)
        logger.info(f"\nSource {i}/{len(results)}")
        logger.info("-" * 40)

        logger.info(f"Distance: {result['distance']:.4f}")

        # Afficher le contenu
        logger.info(result["metadata"].get("content", result["document"])[:200] + "...")
        logger.info("-" * 40)

    logger.info(f"\n📊 Nombre total de sources: {len(results)}")
