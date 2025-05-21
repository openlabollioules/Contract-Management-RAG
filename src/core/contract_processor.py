import time
import re
from typing import List
from pathlib import Path
import hashlib

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
from core.chunk_summarizer import ChunkSummarizer
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def process_contract(filepath: str, summarize_chunks: bool = False) -> List[Chunk]:
    """
    Process a contract file and return intelligent chunks using a semantic approach:
    1. Apply semantic chunking directly on the document
    2. Preserve document metadata
    3. Apply post-processing to restore important legal content that might have been lost
    4. Optionally summarize chunks using Ollama

    Args:
        filepath: Path to the contract file
        summarize_chunks: If True, summarize each chunk using Ollama before adding to the database

    Returns:
        List of Chunk objects with metadata
    """
    logger.info("\n🔄 Début du traitement du document...")
    start_time = time.time()

    # 1. Load and extract text from PDF
    logger.info(
        "📄 Extraction du texte du PDF (avec détection des en-têtes/pieds de page et suppression des références d'images)..."
    )
    text, document_title = extract_pdf_text(filepath)

    logger.info(f"✅ Texte extrait ({len(text.split())} mots)")
    
    # Hashage des mots du document original pour vérification ultérieure
    original_words = set(text.lower().split())
    original_word_count = len(text.split())
    
    logger.info(f"✅ Document original contient {len(original_words)} mots uniques")

    logger.info("\n🔄 Découpage du texte avec approche optimisée pour RAG...")
    
    # Configuration optimale pour RAG - Paramètres optimisés pour les contrats juridiques
    semantic_manager = TextChunker(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",  # Modèle performant pour la sémantique juridique
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.85,  # Seuil plus bas pour créer plus de chunks
        buffer_size=5,     # Buffer plus petit pour plus de précision dans la détection des changements sémantiques
        chunk_size=1500,   # Chunks plus grands pour capturer des clauses complètes
        chunk_overlap=300, # Chevauchement significatif pour éviter les pertes d'information
        preserve_legal_structure=True, # Préserver la structure des clauses juridiques
    )
    
    # Appliquer le découpage sémantique optimisé
    chunks = semantic_manager.chunk_text(text, doc_metadata={"title": document_title})
    
    logger.info(f"✅ Découpage sémantique optimisé pour RAG: {len(chunks)} chunks créés")
    
    # Vérifier que tout le contenu est présent dans les chunks
    all_chunks_text = " ".join([chunk.content for chunk in chunks])
    chunks_words = set(all_chunks_text.lower().split())
    chunks_word_count = len(all_chunks_text.split())
    
    coverage = len(chunks_words.intersection(original_words)) / len(original_words) * 100
    word_count_ratio = chunks_word_count / original_word_count * 100

    logger.info(f"📊 Couverture du texte original: {coverage:.2f}% des mots uniques")
    logger.info(f"📊 Ratio du nombre de mots: {word_count_ratio:.2f}% ({chunks_word_count}/{original_word_count})")
    
    if coverage < 90:
        logger.warning(f"⚠️ Attention: {100-coverage:.2f}% des mots uniques du document original sont absents des chunks!")
        
        # Identifier les mots manquants (limité à 50 pour éviter de surcharger les logs)
        missing_words = list(original_words - chunks_words)
        if missing_words:
            # Filtrer pour ne garder que les mots significatifs (plus de 3 caractères)
            significant_missing = [w for w in missing_words if len(w) > 3]
            if significant_missing:
                logger.warning(f"⚠️ Exemples de mots significatifs manquants: {', '.join(significant_missing[:50])}")
                if len(significant_missing) > 50:
                    logger.warning(f"⚠️ ... et {len(significant_missing) - 50} autres")
    
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
    for i, chunk in enumerate(chunks):
        # Enhanced metadata structure with proper type conversion for ChromaDB
        # Convert any list to string, as ChromaDB doesn't support lists
        hierarchy_str = " -> ".join(chunk.hierarchy) if chunk.hierarchy else "unknown"

        metadata = {
            "section_number": str(chunk.section_number or "unknown"),
            "hierarchy": hierarchy_str,
            "document_title": str(chunk.document_title or "unknown"),
            "parent_section": str(chunk.parent_section or "unknown"),
            "chapter_title": str(chunk.chapter_title or "unknown"),
            "position": str(getattr(chunk, "position", i)),  # Add position as string in metadata
            "total_chunks": str(getattr(chunk, "total_chunks", len(chunks))),  # Add total_chunks as string in metadata
            "content": str(chunk.content),
        }

        # Enhanced content with metadata
        content = f"""
Section: {metadata['section_number']}
Hiérarchie complète: {hierarchy_str}
Document: {metadata['document_title']}
Chapitre: {metadata['chapter_title']}
Position: {metadata['position']} / {metadata['total_chunks']}

Contenu:
{chunk.content}
"""
        chroma_chunks.append({"content": content, "metadata": metadata})

    # 4.5 Optionally summarize chunks
    if summarize_chunks:
        logger.info("\n📝 Résumé des chunks avec Ollama...")
        summarizer = ChunkSummarizer()
        chroma_chunks = summarizer.summarize_chunks(chroma_chunks)
        logger.info("✅ Chunks résumés")

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

    # Print processing time and statistics
    processing_time = time.time() - start_time
    logger.info(f"\n⏱️ Temps total de traitement: {processing_time:.2f} secondes")

    logger.info(f"📊 Nombre de chunks créés: {len(chunks)}")
    logger.info(f"📊 Taille moyenne des chunks: {sum(len(c.content.split()) for c in chunks) / len(chunks):.1f} tokens")
    
    # Exporter tous les chunks pour inspection
    chunks_details_path = f"chunks_details_{Path(filepath).stem}.txt"
    with open(chunks_details_path, 'w') as f:
        f.write(f"Document: {document_title}\n")
        f.write(f"Nombre total de chunks: {len(chunks)}\n")
        f.write(f"Couverture du texte original: {coverage:.2f}%\n")
        f.write(f"Ratio du nombre de mots: {word_count_ratio:.2f}% ({chunks_word_count}/{original_word_count})\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1}/{len(chunks)} ---\n")
            f.write(f"Section: {chunk.section_number or 'Non spécifié'}\n")
            f.write(f"Titre: {chunk.chapter_title or 'Non spécifié'}\n")
            f.write(f"Taille: {len(chunk.content.split())} mots\n")
            f.write("Contenu:\n")
            f.write(chunk.content[:1000])  # Limiter à 1000 caractères pour la lisibilité
            if len(chunk.content) > 1000:
                f.write("...\n")
            f.write("\n\n")
    
    logger.info(f"📄 Détails des chunks exportés dans {chunks_details_path}")

    # Display chunks details and removed content
    display_chunks_details(chunks)
    #display_removed_content(text, chunks)

    return chunks
