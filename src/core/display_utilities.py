import re
from typing import List

from document_processing.contract_splitter import Chunk
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def display_chunks_details(chunks: List[Chunk]) -> None:
    """
    Affiche le contenu détaillé de chaque chunk avec ses métadonnées

    Args:
        chunks: Liste des chunks à afficher
    """
    logger.info("📋 Détails des chunks:")
    logger.debug("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        logger.debug(f"\nChunk {i}/{len(chunks)}")
        logger.debug("-" * 40)
        logger.debug(f"Section: {chunk.section_number}")
        logger.debug(f"Hiérarchie: {' -> '.join(chunk.hierarchy)}")
        logger.debug(f"Document: {chunk.document_title}")
        logger.debug(f"Chapitre: {chunk.chapter_title}")
        logger.debug(f"Section parente: {chunk.parent_section}")
        logger.debug(
            f"Position: {getattr(chunk, 'position', 'N/A')}/{getattr(chunk, 'total_chunks', 'N/A')}"
        )
        logger.debug(f"Taille (mots): {len(chunk.content.split())}")
        
        # Display dates if present
        dates = getattr(chunk, 'dates', [])
        if dates:
            logger.debug(f"Dates détectées: {', '.join(dates)}")
        
        logger.debug("\nContenu:")
        logger.debug(chunk.content)
        logger.debug("-" * 40)


def display_removed_content(original_text: str, chunks: List[Chunk]) -> None:
    """
    Affiche TOUTES les lignes du texte original qui ne sont pas présentes dans les chunks.
    Utilise une comparaison ligne par ligne stricte.

    Args:
        original_text: Texte original du document
        chunks: Liste des chunks après découpage
    """
    logger.info("🗑️ Contenu supprimé lors du découpage:")
    logger.debug("=" * 80)

    # Extraire toutes les lignes du texte original, avec normalisation minimale
    original_lines = []
    for line in original_text.split("\n"):
        line = line.strip()
        if line:  # Ignorer les lignes vides
            original_lines.append(line)

    # Extraire toutes les lignes des chunks
    chunk_lines = []
    for chunk in chunks:
        for line in chunk.content.split("\n"):
            line = line.strip()
            if line:  # Ignorer les lignes vides
                chunk_lines.append(line)

    # Liste très restrictive de patterns qui correspondent uniquement à des titres
    # Ces patterns doivent être strictement limités aux formats de titres courants dans les contrats
    strict_title_patterns = [
        # Titres markdown (# Titre)
        r"^#+\s+\**[A-Z][A-Z\s]+\**$",
        # Formats explicites de sections numérotées
        r"^ARTICLE\s+[IVXLCDM]+\s*[:\.]",
        r"^SECTION\s+[0-9]+\s*[:\.]",
        r"^CHAPTER\s+[0-9]+\s*[:\.]",
        r"^ANNEXE?\s+[A-Z]:\s*",
        r"^APPENDIX\s+[A-Z]:\s*",
        # Titres numérotés tout en majuscules (très spécifique)
        r"^[0-9]+\.\s+[A-Z][A-Z\s]+$",
        # Nouveaux patterns plus précis pour les titres de contrats
        r"^#+\s+\**\d+\.\d+\s+[A-Z][A-Z\s]+\**$",  # ## 1.2 TITLE
        r"^#+\s+\**\d+\.\d+\.\d+\s+[A-Z][A-Z\s]+\**$",  # ## 1.2.3 TITLE
        r"^\*\*[A-Z][A-Z\s]+\*\*$",  # **TITLE**
        r"^[A-Z][A-Z\s]+:$",  # TITLE:
    ]

    logger.debug(f"Nombre de patterns de titre: {len(strict_title_patterns)}")

    # Fonction pour vérifier si une ligne est un titre
    def is_strict_title(line):
        # Vérifier les patterns explicites de titre
        for pattern in strict_title_patterns:
            if re.match(pattern, line):
                logger.debug(f"Titre trouvé avec pattern {pattern}: {line}")
                return True

        # Vérifier le cas spécial pour "# N. TITLE"
        if re.match(r"^#+\s+\**[0-9]+\.\s+[A-Z][A-Z\s]+\**$", line):
            logger.debug(f"Titre spécial trouvé: {line}")
            return True

        # Si le texte est court (<5 mots), tout en majuscules et pas de ponctuation finale,
        # c'est probablement un titre
        words = line.split()
        if (
            len(words) <= 5
            and line.isupper()
            and not line.endswith((".", "!", "?", ",", ";", ":", ")", "]"))
        ):
            logger.debug(f"Titre court en majuscules trouvé: {line}")
            return True

        return False

    # Comparer chaque ligne originale avec les lignes des chunks
    # Une ligne est considérée présente si elle est exactement dans les chunks (ou très légèrement différente)
    def is_line_present(line, chunk_lines):
        # Normalisation minimale
        def normalize_for_comparison(text):
            # Supprime juste les espaces en début/fin et réduit les espaces multiples
            return re.sub(r"\s+", " ", text).strip()

        normalized_line = normalize_for_comparison(line)

        # Vérification exacte
        for chunk_line in chunk_lines:
            normalized_chunk_line = normalize_for_comparison(chunk_line)
            if normalized_line == normalized_chunk_line:
                return True

            # Vérification avec légère tolérance pour les espaces/tirets/points
            # Remplacer les caractères spéciaux par des espaces et comparer
            clean_line = re.sub(r"[-_.,;:()]", " ", normalized_line)
            clean_line = re.sub(r"\s+", " ", clean_line).strip()

            clean_chunk = re.sub(r"[-_.,;:()]", " ", normalized_chunk_line)
            clean_chunk = re.sub(r"\s+", " ", clean_chunk).strip()

            if clean_line == clean_chunk:
                return True

        return False

    # Trouver les lignes qui ne sont pas dans les chunks
    missing_titles = []
    missing_content = []

    logger.debug(f"Nombre de lignes originales: {len(original_lines)}")
    logger.debug(f"Nombre de lignes dans les chunks: {len(chunk_lines)}")

    for i, line in enumerate(original_lines):
        if not is_line_present(line, chunk_lines):
            # Collecter le contexte (ligne précédente et suivante)
            context = []
            if i > 0:
                context.append(f"Ligne précédente: {original_lines[i-1]}")

            if i < len(original_lines) - 1:
                context.append(f"Ligne suivante: {original_lines[i+1]}")

            # Vérifier si c'est un titre
            if is_strict_title(line):
                context.append(f"TITRE supprimé: {line}")
                missing_titles.append((context, line))
            else:
                context.append(f"LIGNE supprimée: {line}")
                missing_content.append((context, line))

    # Afficher les titres supprimés
    if missing_titles:
        logger.warning(f"Nombre de titres supprimés: {len(missing_titles)}")
        logger.debug("\n📑 Titres supprimés:")
        logger.debug("-" * 40)
        for context, title in missing_titles:
            for line in context:
                logger.debug(f"- {line}")
            logger.debug("-" * 40)

    # Afficher les lignes de contenu supprimées
    if missing_content:
        logger.warning(f"\n📄 {len(missing_content)} lignes supprimées")
        logger.debug("\n⚠️ Détail des lignes supprimées:")
        logger.debug("-" * 40)
        for context, content in missing_content:
            for line in context:
                logger.debug(f"- {line}")
            logger.debug("-" * 40)

    # Statistiques
    logger.info(f"\n📊 Statistiques du traitement:")
    logger.info(f"- Nombre total de titres supprimés: {len(missing_titles)}")
    logger.info(f"- Nombre total de lignes supprimées: {len(missing_content)}")

    if not missing_titles and not missing_content:
        logger.info("Aucune ligne n'a été supprimée lors du découpage.")


def display_semantic_split_chunks(
    structure_chunks: List[Chunk], final_chunks: List[Chunk]
) -> None:
    """
    Affiche les chunks qui ont subi un découpage sémantique dans l'approche hybride

    Args:
        structure_chunks: Chunks initiaux après découpage structurel
        final_chunks: Chunks finaux après découpage sémantique
    """
    logger.info("\n🔍 Chunks ayant subi un découpage sémantique:")
    logger.debug("=" * 80)

    # Identifier les chunks originaux qui ont été découpés
    split_chunks = []
    for original_chunk in structure_chunks:
        # Compter combien de chunks finaux proviennent de ce chunk original
        sub_chunks = [
            c for c in final_chunks if c.section_number == original_chunk.section_number
        ]
        if len(sub_chunks) > 1:  # Si le chunk a été découpé
            split_chunks.append((original_chunk, sub_chunks))

    if not split_chunks:
        logger.info("Aucun chunk n'a subi de découpage sémantique.")
        return

    logger.info(f"Nombre de chunks découpés sémantiquement: {len(split_chunks)}")

    for original_chunk, sub_chunks in split_chunks:
        logger.debug("\n" + "=" * 40)
        logger.debug(f"Chunk original:")
        logger.debug(f"Section: {original_chunk.section_number}")
        logger.debug(f"Hiérarchie: {' -> '.join(original_chunk.hierarchy)}")
        logger.debug(f"Taille originale: {len(original_chunk.content.split())} mots")
        logger.debug("\nDécoupé en {len(sub_chunks)} sous-chunks:")

        for i, sub_chunk in enumerate(sub_chunks, 1):
            logger.debug(f"\nSous-chunk {i}/{len(sub_chunks)}:")
            logger.debug(
                f"Position: {getattr(sub_chunk, 'position', 'N/A')}/{getattr(sub_chunk, 'total_chunks', 'N/A')}"
            )
            logger.debug(f"Taille: {len(sub_chunk.content.split())} mots")
            logger.debug(
                f"Contenu: {sub_chunk.content[:200]}..."
            )  # Afficher les 200 premiers caractères
        logger.debug("=" * 40)
