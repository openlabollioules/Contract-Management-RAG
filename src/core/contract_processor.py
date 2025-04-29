import time
from typing import List
from pathlib import Path

from core.content_restoration import restore_important_content
from core.display_utilities import (display_chunks_details,
                                    display_removed_content,
                                    display_semantic_split_chunks)
from document_processing.contract_splitter import Chunk, ContractSplitter
from document_processing.pdf_extractor import extract_pdf_text
from document_processing.text_chunker import TextChunker
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from core.graph_manager import GraphManager
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def process_contract(filepath: str) -> List[Chunk]:
    """
    Process a contract file and return intelligent chunks using a hybrid approach:
    1. First split by legal structure (articles, sections, subsections)
    2. Then apply semantic chunking for sections exceeding 800 tokens
    3. Preserve hierarchical metadata for traceability
    4. Apply post-processing to restore important legal content that might have been lost

    Args:
        filepath: Path to the contract file

    Returns:
        List of Chunk objects with preserved legal structure and metadata
    """
    logger.info("\n🔄 Début du traitement du document...")
    start_time = time.time()

    # 1. Load and extract text from PDF
    logger.info(
        "📄 Extraction du texte du PDF (avec détection des en-têtes/pieds de page et suppression des références d'images)..."
    )
    text, document_title = extract_pdf_text(filepath)
    logger.info(f"✅ Texte extrait ({len(text.split())} mots)")

    logger.info(
        "\n🔄 Découpage du texte avec approche hybride (structure + sémantique)..."
    )
    # First split by legal structure
    splitter = ContractSplitter(document_title=document_title)
    structure_chunks = splitter.split(text)

    # Then apply semantic chunking for large sections
    semantic_manager = TextChunker(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.6,
        buffer_size=3,
        chunk_size=800,  # Limite de ~800 tokens
        chunk_overlap=100,  # Chevauchement de 100 tokens
    )

    chunks = []
    for chunk in structure_chunks:
        # If section is small enough, keep it as is
        if len(chunk.content.split()) < 800:  # Approximate token count
            chunks.append(chunk)
        else:
            # For large sections, apply semantic chunking
            sub_chunks = semantic_manager.chunk_text(chunk.content)
            # Preserve metadata in sub-chunks
            for sub_chunk in sub_chunks:
                sub_chunk.section_number = chunk.section_number
                sub_chunk.hierarchy = chunk.hierarchy
                sub_chunk.document_title = chunk.document_title
                sub_chunk.parent_section = chunk.parent_section
                sub_chunk.chapter_title = chunk.chapter_title
                # Add position metadata
                sub_chunk.position = len(chunks)
                sub_chunk.total_chunks = len(sub_chunks)
            chunks.extend(sub_chunks)

    # 2.5 Post-traitement: restaurer le contenu juridique important qui aurait pu être perdu
    logger.info(
        "\n🔄 Application du post-traitement pour restaurer le contenu juridique important..."
    )
    chunks = restore_important_content(text, chunks)

    # 3. Initialize embeddings and ChromaDB
    logger.info("\n🔍 Initialisation des embeddings et de ChromaDB...")
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)

    # 4. Prepare chunks for ChromaDB with enhanced metadata
    logger.info("\n📦 Préparation des chunks pour ChromaDB...")
    chroma_chunks = []
    for chunk in chunks:
        # Enhanced metadata structure with proper type conversion for ChromaDB
        # Convert any list to string, as ChromaDB doesn't support lists
        hierarchy_str = " -> ".join(chunk.hierarchy) if chunk.hierarchy else "unknown"

        metadata = {
            "section_number": str(chunk.section_number or "unknown"),
            "hierarchy": hierarchy_str,
            "document_title": str(chunk.document_title or "unknown"),
            "parent_section": str(chunk.parent_section or "unknown"),
            "chapter_title": str(chunk.chapter_title or "unknown"),
            "title": str(document_title),
            "content": str(chunk.content),
            "chunk_type": str(getattr(chunk, "chunk_type", "unknown")),
            "position": str(getattr(chunk, "position", "0")),
            "total_chunks": str(getattr(chunk, "total_chunks", "0")),
            "chunk_size": str(len(chunk.content.split())),  # Approximate token count
            "timestamp": str(time.time()),
        }

        # Enhanced content with metadata
        content = f"""
Section: {metadata['section_number']}
Hiérarchie complète: {hierarchy_str}
Document: {metadata['document_title']}
Position: {metadata['position']}/{metadata['total_chunks']}

Contenu:
{chunk.content}
"""
        chroma_chunks.append({"content": content, "metadata": metadata})

    # 5. Add chunks to ChromaDB
    logger.info("\n📦 Ajout des chunks à ChromaDB...")
    chroma_manager.add_documents(chroma_chunks)
    logger.info("✅ Chunks ajoutés à ChromaDB")

    logger.info("🔄 Building knowledge graph...")
    graph_manager = GraphManager(chroma_manager, embeddings_manager)
    graph = graph_manager.build_graph(chroma_chunks)

    # logger.info("🎨 Generating graph visualization...")
    # graph_output_path = f"graph_{Path(filepath).stem}.png"
    # graph_manager.visualize_graph(graph, output_path=graph_output_path)

    logger.info("📝 Exporting graph details...")
    details_output_path = f"graph_details_{Path(filepath).stem}.txt"
    graph_manager.export_graph_details(graph, output_path=details_output_path)

    # Print document metadata
    logger.info("\nDocument Metadata:")
    logger.info(f"- Title: {document_title}")
    logger.info(f"- Author: Unknown")
    logger.info(f"- Pages: Unknown")

    # Print processing time and statistics
    processing_time = time.time() - start_time
    logger.info(f"\n⏱️ Temps total de traitement: {processing_time:.2f} secondes")

    logger.info(f"📊 Nombre de chunks créés: {len(chunks)}")
    logger.info(
        f"📊 Taille moyenne des chunks: {sum(len(c.content.split()) for c in chunks) / len(chunks):.1f} tokens"
    )

    # Display chunks details and removed content
    display_chunks_details(chunks)
    display_removed_content(text, chunks)

    # Display semantic split chunks if in hybrid mode
    if structure_chunks:
        display_semantic_split_chunks(structure_chunks, chunks)

    return chunks
