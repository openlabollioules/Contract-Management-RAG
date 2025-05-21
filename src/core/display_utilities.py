import re
from typing import List

from document_processing.contract_splitter import Chunk
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def display_chunks_details(chunks: List[Chunk]) -> None:
    """
    Affiche les détails des chunks pour inspection.

    Args:
        chunks: Liste des chunks à afficher
    """
    logger.info("📋 Détails des chunks:")
    logger.debug("=" * 80)

    for i, chunk in enumerate(chunks):
        logger.debug(f"\nChunk {i+1}/{len(chunks)}")
        logger.debug("-" * 40)
        logger.debug(f"Section: {chunk.section_number or 'None'}")
        # Correction pour gérer le cas où hierarchy est None
        if chunk.hierarchy:
            hierarchy_str = " -> ".join(chunk.hierarchy)
            logger.debug(f"Hiérarchie: {hierarchy_str}")
        else:
            logger.debug("Hiérarchie: Non définie")
        logger.debug(f"Document: {chunk.document_title or 'Non spécifié'}")
        logger.debug(f"Chapitre: {chunk.chapter_title or 'Non spécifié'}")
        logger.debug(f"Section parente: {chunk.parent_section or 'None'}")

        # Position info if available
        if hasattr(chunk, "position") and hasattr(chunk, "total_chunks"):
            logger.debug(f"Position: {chunk.position}/{chunk.total_chunks}")

        # Additional info if available
        if hasattr(chunk, "token_count"):
            logger.debug(f"Taille (mots): {chunk.token_count}")

        # Display references if available
        if hasattr(chunk, "references") and chunk.references:
            logger.debug(f"Références: {', '.join(chunk.references)}")

        # Display dates if available
        if hasattr(chunk, "dates") and chunk.dates:
            logger.debug(f"Dates détectées: {', '.join(chunk.dates)}")
            
        logger.debug("\nContenu:")
        logger.debug(chunk.content[:1500] + ("..." if len(chunk.content) > 1500 else ""))
        logger.debug("-" * 40)


def display_removed_content(full_text: str, chunks: List[Chunk]) -> None:
    """
    Compare le texte original avec le contenu des chunks pour identifier
    le contenu potentiellement supprimé ou non inclus.

    Args:
        full_text: Texte complet du document
        chunks: Liste des chunks générés
    """
    logger.info("\n🔍 Analyse du contenu supprimé ou modifié:")

    # Texte des chunks combinés
    combined_chunks_text = " ".join([chunk.content for chunk in chunks])

    # Normaliser les textes pour la comparaison (supprimer espaces multiples, tabs, etc.)
    full_text_normalized = re.sub(r'\s+', ' ', full_text).strip()
    chunks_text_normalized = re.sub(r'\s+', ' ', combined_chunks_text).strip()

    # Tokenisation simple des textes (division en mots)
    full_text_words = set(full_text_normalized.lower().split())
    chunks_words = set(chunks_text_normalized.lower().split())

    # Mots présents dans le texte original mais absents des chunks
    missing_words = full_text_words - chunks_words

    # Calcul du pourcentage de contenu préservé
    if len(full_text_words) > 0:
        preservation_percentage = (len(full_text_words.intersection(chunks_words)) / len(full_text_words)) * 100
    else:
        preservation_percentage = 100.0

    logger.info(f"Pourcentage de préservation: {preservation_percentage:.2f}%")
    logger.info(f"Nombre de mots manquants: {len(missing_words)} sur {len(full_text_words)}")

    # Afficher un échantillon de mots manquants (significatifs)
    significant_missing = [w for w in missing_words if len(w) > 3]
    if significant_missing:
        sample = ', '.join(sorted(significant_missing)[:20])
        logger.info(f"Échantillon de mots manquants: {sample}...")


def display_semantic_split_chunks(chunks: List[str]) -> None:
    """
    Affiche les chunks créés par division sémantique pour inspection.

    Args:
        chunks: Liste des chunks sémantiques à afficher
    """
    logger.info("\n📋 Chunks sémantiques:")
    
    for i, chunk in enumerate(chunks):
        logger.debug(f"\nChunk sémantique {i+1}/{len(chunks)}")
        logger.debug("-" * 40)
        logger.debug(chunk[:1000] + ("..." if len(chunk) > 1000 else ""))
        logger.debug("-" * 40)
