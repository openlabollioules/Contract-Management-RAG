import sys
import time
from typing import List

from rag.chroma_manager import ChromaDBManager
from rag.embeddings_manager import EmbeddingsManager
from rag.hierarchical_grouper import HierarchicalGrouper
from rag.intelligent_splitter import Chunk, IntelligentSplitter
from rag.pdf_loader import extract_text_contract
from rag.ollama_manager import OllamaManager


def process_contract(filepath: str) -> List[Chunk]:
    """
    Process a contract file and return intelligent chunks

    Args:
        filepath: Path to the contract file

    Returns:
        List of Chunk objects
    """
    print("\n🔄 Début du traitement du document...")
    start_time = time.time()

    # 1. Load and extract text from PDF
    print("📄 Extraction du texte du PDF...")
    text, document_title = extract_text_contract(filepath)
    print(f"✅ Texte extrait ({len(text.split())} mots)")

    # 2. Split text into intelligent chunks
    print("\n🔍 Découpage du texte en chunks intelligents...")
    splitter = IntelligentSplitter(document_title=document_title)
    chunks = splitter.split(text)

    # 3. Group chunks hierarchically
    print("\n🔍 Regroupement hiérarchique des chunks...")
    grouper = HierarchicalGrouper()
    hierarchical_groups = grouper.group_chunks(chunks)

    # 4. Initialize embeddings and ChromaDB
    print("\n🔍 Initialisation des embeddings et de ChromaDB...")
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

    # 5. Prepare chunks for ChromaDB
    print("\n📦 Préparation des chunks pour ChromaDB...")
    chroma_chunks = []
    for chunk in chunks:
        # Préparer les métadonnées avec la même structure que display_chunks
        metadata = {
            "section_number": chunk.section_number or "unknown",
            "hierarchy": chunk.hierarchy or ["unknown"],
            "document_title": chunk.document_title or "unknown",
            "parent_section": chunk.parent_section or "unknown",
            "chapter_title": chunk.chapter_title or "unknown",
            "title": document_title,
            "content": chunk.content,  # Stocker le contenu brut
            "chunk_type": str(chunk.chunk_type) if hasattr(chunk, 'chunk_type') else "unknown"
        }

        # Ajouter le contenu avec les métadonnées
        content = f"""
Section: {metadata['section_number']}
Hiérarchie complète: {' -> '.join(metadata['hierarchy'])}
Document: {metadata['document_title']}

Contenu:
{chunk.content}
"""
        chroma_chunks.append({"content": content, "metadata": metadata})

    # 6. Add chunks to ChromaDB
    print("\n💾 Ajout des chunks à ChromaDB...")
    chroma_manager.add_documents(chroma_chunks)
    print("✅ Chunks ajoutés à ChromaDB")

    # Print document metadata
    print("\nDocument Metadata:")
    print(f"- Title: {document_title}")
    print(f"- Author: Unknown")
    print(f"- Pages: Unknown")

    return chunks


def search_contracts(query: str, n_results: int = 5) -> None:
    """
    Search in the contract database

    Args:
        query: Search query
        n_results: Number of results to return
    """
    print(f"\n🔍 Recherche: {query}")

    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

    # Search
    results = chroma_manager.search(query, n_results=n_results)

    # Display results
    print(f"\n📊 Résultats ({len(results)} trouvés):")
    for i, result in enumerate(results, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Section: {result['metadata']['section']}")
        print(f"Hiérarchie: {result['metadata']['hierarchy']}")
        print(f"Document: {result['metadata']['document_title']}")
        print(f"Contenu: {result['document'][:200]}...")
        print(f"Distance: {result['distance']:.4f}")


def chat_with_contract(query: str, n_context: int = 3) -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
    """
    print(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

    # Search for relevant context
    results = chroma_manager.search(query, n_results=n_context)

    # Prepare context for the prompt
    context = "\n\n".join([
        f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
        f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
        f"Chapter: {result['metadata'].get('chapter_title', 'Non spécifié')}\n"
        f"Content: {result['document']}"
        for result in results
    ])

    # Create the prompt with context
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats. 
Voici le contexte pertinent extrait des documents :

{context}

Question de l'utilisateur : {query}

Réponds de manière précise en te basant uniquement sur le contexte fourni. 
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""

    # Get response from Ollama
    from rag.ollama_chat import ask_ollama
    response = ask_ollama(prompt)
    print("\n🤖 Réponse :")
    print(response)

    # Display sources with metadata
    print("\n📚 Sources :")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print("\n" + "-" * 40)
        print(f"\nSource {i}/{len(results)}")
        print("-" * 40)
            
        print(f"Distance: {result['distance']:.4f}")
        
        # Afficher le contenu
        print(result['metadata'].get('content', result['document'])[:200] + "...")
        print("-" * 40)

    print(f"\n📊 Nombre total de sources: {len(results)}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <contract_file> [search_query|--chat]")
        sys.exit(1)

    filepath = sys.argv[1]

    # If --chat is provided, enter chat mode
    if len(sys.argv) > 2 and sys.argv[2] == "--chat":
        print("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nVotre question : ")
            if query.lower() == 'exit':
                break
            chat_with_contract(query)
    # If search query is provided, perform search
    elif len(sys.argv) > 2:
        search_query = " ".join(sys.argv[2:])
        search_contracts(search_query)
    else:
        # Process the contract
        chunks = process_contract(filepath)
