import re

from document_processing.contract_splitter import ContractSplitter
from document_processing.text_chunker import TextChunker
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def hybrid_chunk_text(text, document_title):
    """
    Méthode hybride combinant chunking intelligent et sémantique

    Args:
        text: Texte à découper
        document_title: Titre du document

    Returns:
        Liste des chunks finaux
    """
    # 1. Découper d'abord selon la structure (sections principales)
    splitter = ContractSplitter(document_title=document_title)
    structure_chunks = splitter.split(text)

    # 2. Pour chaque grande section, appliquer le chunking sémantique si nécessaire
    final_chunks = []
    for chunk in structure_chunks:
        # Si le chunk est petit, le garder tel quel
        if len(chunk.content) < 1000:
            final_chunks.append(chunk)
        else:
            # Pour les sections longues, appliquer chunking sémantique
            semantic_manager = TextChunker(
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.6,
                buffer_size=2,
            )
            # Préserver les métadonnées de la section dans les sous-chunks
            semantic_sub_chunks = semantic_manager.chunk_text(chunk.content)
            for sub_chunk in semantic_sub_chunks:
                sub_chunk.section_number = chunk.section_number
                sub_chunk.hierarchy = chunk.hierarchy
                sub_chunk.document_title = chunk.document_title
                sub_chunk.parent_section = chunk.parent_section
                sub_chunk.chapter_title = chunk.chapter_title
            final_chunks.extend(semantic_sub_chunks)

    return final_chunks


def calculate_optimal_threshold(text):
    """
    Calcule le seuil optimal basé sur la complexité du texte

    Args:
        text: Texte à analyser

    Returns:
        Seuil optimal calculé
    """
    # Cette fonction est maintenant déléguée au TextChunker qui contient
    # une implémentation plus sophistiquée via _calculate_optimal_threshold

    from document_processing.text_chunker import TextChunker

    # Créer une instance temporaire pour utiliser la méthode
    temp_manager = TextChunker()

    # Utiliser la méthode améliorée du manager
    return temp_manager._calculate_optimal_threshold(text)


def preprocess_legal_text(text):
    """
    Prétraite le texte juridique pour préserver les clauses

    Args:
        text: Texte juridique à prétraiter

    Returns:
        Texte prétraité avec marqueurs de clauses et références croisées
    """
    # Cette fonction est maintenant déléguée au TextChunker qui contient
    # une implémentation plus sophistiquée via _preprocess_text_with_section_markers
    # avec un ensemble élargi de patterns juridiques et de références croisées

    # Nous gardons ce code pour compatibilité, mais il utilise maintenant les patterns
    # du TextChunker pour une cohérence dans le traitement

    from document_processing.text_chunker import TextChunker

    # Créer une instance temporaire pour accéder aux patterns
    temp_manager = TextChunker()

    # Traiter le texte en utilisant les patterns du manager
    processed_lines = []
    for line in text.split("\n"):
        if any(re.search(pattern, line) for pattern in temp_manager.legal_patterns):
            processed_lines.append("[CLAUSE_START]" + line)
        elif any(
            re.search(pattern, line) for pattern in temp_manager.cross_ref_patterns
        ):
            processed_lines.append("[CROSS_REF]" + line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)
