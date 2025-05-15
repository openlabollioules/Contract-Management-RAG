import re
from typing import List

from document_processing.contract_splitter import Chunk
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def restore_important_content(original_text: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    Fonction de post-traitement qui identifie les lignes juridiques importantes
    qui ont été supprimées et les réintègre dans les chunks appropriés.
    NE restaure PAS les titres, mais restaure toutes les lignes de contenu juridique.

    Args:
        original_text: Texte original du document
        chunks: Liste des chunks après découpage initial

    Returns:
        Liste des chunks avec le contenu important restauré
    """
    logger.info(
        "\n🔄 Post-traitement: recherche de contenu juridique important supprimé..."
    )

    # Extraire toutes les lignes du texte original
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

    # Fonction pour normaliser les lignes avant comparaison
    def normalize_line(text):
        return re.sub(r"\s+", " ", text).strip()

    # Identifier les lignes manquantes
    missing_lines = []
    for line in original_lines:
        normalized_line = normalize_line(line)
        found = False

        for chunk_line in chunk_lines:
            normalized_chunk = normalize_line(chunk_line)
            if normalized_line == normalized_chunk:
                found = True
                break

        if not found:
            missing_lines.append(line)

    # Fonction pour détecter si une ligne est un titre
    def is_title(line):
        # Exception pour les sections avec format markdown qui contiennent aussi du contenu juridique
        # comme "#### 13.3.1 For each Unit, Seller shall achieve..."
        if re.match(r"^#{1,6}\s+\d+(\.\d+)*\s+.*\b(shall|may|must|will)\b", line):
            return False
            
        # Patterns pour identifier les titres
        title_patterns = [
            # Titres markdown (# Titre)
            r"^#+\s+\**[A-Za-z0-9]",
            # Formats explicites de sections numérotées
            r"^ARTICLE\s+[IVXLCDM]+\s*[:\.]",
            r"^SECTION\s+[0-9]+\s*[:\.]",
            r"^CHAPTER\s+[0-9]+\s*[:\.]",
            r"^ANNEXE?\s+[A-Z]:",
            r"^APPENDIX\s+[A-Z]:",
            # Titres numérotés
            r"^[0-9]+(\.[0-9]+)*\s+[A-Z]",
            # Titres avec des caractères spéciaux
            r"^\*\*[A-Z]",
            r"^[A-Z][A-Z\s]+:$",
            # Titres courts tout en majuscules
            r"^[A-Z][A-Z\s]{1,30}$",
            # Titres de sections Markdown
            r"^#{1,6}\s+.*$",
            # Identifiants de section explicites
            r"^(ARTICLE|Section|Chapter|§)\s+\d+(\.\d+)*",
        ]

        # Vérifier si la ligne correspond à un des patterns de titre
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True

        # Autres indices de titres
        if line.isupper() and len(line.split()) <= 5:
            return True

        # Vérifier spécifiquement pour les titres de section markdown
        if re.match(r"^#{1,6}\s+.*$", line) or re.match(r"^#{1,6}$", line):
            return True

        # Vérifier pour les titres de style "1.1 Definitions"
        if re.match(r"^(\d+\.)+\d*\s+[A-Z][a-z]+.*$", line):
            return True

        return False

    # Critères pour identifier les lignes juridiques importantes - APPROCHE TRÈS PERMISSIVE
    def is_important_legal_content(line):
        # D'abord vérifier si c'est un titre - si oui, ce n'est pas du contenu juridique à restaurer
        if is_title(line):
            # Exception: Si la ligne commence par une lettre/nombre entre parenthèses comme (a), (b), etc.,
            # ou des formats comme a), b) - c'est probablement une liste d'items juridiques, pas un titre
            if re.match(r"^\([a-zA-Z0-9]\)|^[a-zA-Z0-9]\)", line):
                return True
            return False

        # Format de liste numérotée qui doit être considéré comme contenu juridique
        if re.match(r"^\([a-zA-Z0-9]\)|^[a-zA-Z0-9]\)", line):
            return True

        # Si la ligne commence par "Whereas, " c'est probablement un préambule de contrat à conserver
        if line.startswith("Whereas,") or line.startswith("WHEREAS,"):
            return True
            
        # Si la ligne commence par "for the " suivi d'un nom, c'est probablement un contenu important
        if re.match(r"^for the [A-Za-z]", line):
            return True
            
        # Détecter les paragraphes de texte simple (non-titres) de longueur significative
        if len(line.split()) > 5 and not line.startswith('#') and not re.match(r"^\d+(\.\d+)*\s", line):
            # Vérifier si c'est un paragraphe normal et non un titre
            if not line.isupper() and not line.endswith(':'):
                return True

        # Vérifier si la ligne correspond à un format de section Markdown (####) suivi de texte
        if re.match(r"^#{1,6}\s+\d+(\.\d+)*\s", line):
            return True
            
        # Spécifiquement pour restaurer les sections avec format markdown qui contiennent du contenu juridique
        # comme '#### 13.3.1 For each Unit, Seller shall achieve...'
        if re.match(r"^#{1,6}\s+\d+(\.\d+)*\s+.*\b(shall|may|must|will)\b", line):
            return True
            
        # Vérifier les lignes courtes avec des mots-clés juridiques importants
        if (len(line.split()) <= 5 and any(x in line.lower() for x in ["date", "signature", "between", "and", "list", "exhibit", "therefore", "contract", "shall", "mean", "defined", "pursuant", "under", "according", "damage", "payment"])):
            return True

        # Mots-clés juridiques importants (LISTE ÉTENDUE)
        legal_keywords = [
            # Termes juridiques standards
            "notwithstanding",
            "shall be",
            "exclusive remedy",
            "sole and exclusive",
            "right to terminate",
            "limitation of liability",
            "indemnify",
            "warranty",
            "warranties",
            "liabilities",
            "liability",
            "remedies",
            "remedy",
            "disclaims",
            "disclaim",
            "claims",
            "claim",
            "damages",
            "damage",
            "breach",
            "termination",
            "terminate",
            "force majeure",
            "intellectual property",
            "confidential",
            "liquidated damages",
            "penalties",
            "penalty",
            # Termes contractuels additionnels
            "shall",
            "obligation",
            "obligations",
            "responsibility",
            "responsibilities",
            "rights",
            "right",
            "terms",
            "conditions",
            "provisions",
            "stipulations",
            "agreement",
            "contract",
            "hereof",
            "herein",
            "thereof",
            "therein",
            "delivery",
            "deliver",
            "payment",
            "pay",
            "price",
            "fee",
            "fees",
            "delay",
            "delays",
            "timely",
            "schedule",
            "schedules",
            "deadline",
            "deadline",
            "milestones",
            "milestone",
            "completion",
            "complete",
            "acceptance",
            "accepts",
            "accept",
            "approved",
            "approve",
            "approval",
            "rejected",
            "reject",
            "rejection",
            "dispute",
            "disputes",
            "resolution",
            "test",
            "testing",
            "inspection",
            "inspect",
            "audit",
            "review",
            "pursuant",
            "accordance",
            "compliance",
            "comply",
            "applicable",
            "indemnification",
            "indemnify",
            "indemnified",
            "indemnities",
            "insurance",
            "insured",
            "coverage",
            "purchaser",
            "supplier",
            "parties",
        ]

        # Expressions régulières pour clauses spécifiques
        clause_references = [
            r"clause\s+\d+(\.\d+)?",
            r"article\s+\d+(\.\d+)?",
            r"section\s+\d+(\.\d+)?",
            r"pursuant to",
            r"in accordance with",
            r"subject to",
            r"appendix [a-z]",
            r"annex [a-z]",
        ]

        # Vérifier les mots-clés
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in legal_keywords):
            return True

        # Vérifier les références à des clauses
        if any(re.search(pattern, line_lower) for pattern in clause_references):
            return True

        # Reconnaître les clauses de limitation ou d'exclusion
        exclusion_patterns = [
            r"not be liable",
            r"no liability",
            r"shall not",
            r"exclude[sd]?",
            r"except",
            r"exempted",
            r"limitation",
            r"limited to",
            r"restrict(ed|ion)",
            r"waive[sd]?",
            r"waiver",
        ]

        if any(re.search(pattern, line_lower) for pattern in exclusion_patterns):
            return True

        # Structure conditionnelle typique des clauses contractuelles
        if re.search(r"if\s+.*\s+(shall|may|must|will|is|are)\s+", line_lower):
            return True

        # Si la ligne contient des termes d'obligations, de conditions ou de conséquences
        if re.search(
            r"(shall|may|must|will)\s+.*\s+(if|unless|until|provided that)", line_lower
        ):
            return True

        # Lignes qui commencent par des termes d'obligation contractuelle
        if re.search(
            r"^(the\s+)?(purchaser|supplier|contractor|client|party|parties)\s+(shall|may|will|must)",
            line_lower,
        ):
            return True

        # Lignes qui parlent de documents, livraisons, ou paiements
        if re.search(
            r"(documents?|delivery|payment|invoice|fee|compensation|reimbursement)",
            line_lower,
        ):
            return True

        return False

    # Identifier les lignes juridiques importantes parmi les lignes manquantes
    important_lines = [
        line for line in missing_lines if is_important_legal_content(line)
    ]

    if not important_lines:
        logger.info("✅ Aucun contenu juridique important n'a été supprimé.")
        return chunks

    logger.info(
        f"🔍 {len(important_lines)} lignes de contenu juridique important identifiées pour restauration."
    )

    # Fonction pour trouver le meilleur chunk pour restaurer une ligne
    def find_best_chunk(line, chunks):
        # Trouver le contexte de la ligne dans le texte original
        line_index = original_lines.index(line)
        context_before = original_lines[max(0, line_index - 5) : line_index]
        context_after = original_lines[
            line_index + 1 : min(len(original_lines), line_index + 6)
        ]

        best_chunk = None
        best_score = -1

        for chunk in chunks:
            score = 0
            chunk_content = chunk.content.lower()

            # Vérifier si des lignes du contexte sont dans ce chunk
            for ctx_line in context_before + context_after:
                if normalize_line(ctx_line.lower()) in normalize_line(chunk_content):
                    score += 3  # Augmenter le poids du contexte

            # Vérifier si le chunk contient des mots-clés de la même section
            line_words = set(line.lower().split())
            chunk_words = set(chunk_content.split())
            common_words = line_words.intersection(chunk_words)
            score += len(common_words) * 0.2

            # Vérifier si le numéro de section correspond
            if hasattr(chunk, "section_number") and chunk.section_number:
                # Extraire des numéros potentiels de section depuis la ligne
                section_matches = re.findall(r"clause\s+(\d+(\.\d+)?)", line.lower())
                if section_matches:
                    for match in section_matches:
                        if match[0] in chunk.section_number:
                            score += 3

            if score > best_score:
                best_score = score
                best_chunk = chunk

        # Si aucun chunk n'a un bon score, prendre celui qui a la meilleure correspondance textuelle
        if best_score <= 1:
            highest_score = -1
            best_matching_chunk = None

            for chunk in chunks:
                chunk_content = chunk.content.lower()

                # Si la ligne fait partie d'une section numérotée, essayer de trouver cette section
                section_match = re.search(r"\b(\d+(\.\d+)?)\b", line.lower())
                if section_match and section_match.group(1) in chunk_content:
                    return chunk

                # Sinon, utiliser la correspondance textuelle
                line_words = set(line.lower().split())
                chunk_words = set(chunk_content.split())
                common_words = line_words.intersection(chunk_words)

                if len(line_words) > 0:
                    score = len(common_words) / len(line_words)

                    if score > highest_score:
                        highest_score = score
                        best_matching_chunk = chunk

            if best_matching_chunk:
                return best_matching_chunk

        return best_chunk

    # Restaurer les lignes importantes dans les chunks appropriés
    restored_chunks = list(chunks)  # Copie pour éviter de modifier l'original
    for line in important_lines:
        best_chunk = find_best_chunk(line, restored_chunks)
        if best_chunk:
            # Ajouter la ligne au chunk (à la fin)
            best_chunk.content = best_chunk.content + "\n\n" + line
            logger.info(f"✅ Ligne restaurée dans un chunk approprié: {line[:60]}...")
        else:
            # Si aucun chunk approprié n'est trouvé, créer un nouveau chunk
            logger.warning(
                f"⚠️ Création d'un nouveau chunk pour la ligne: {line[:60]}..."
            )
            new_chunk = Chunk(
                content=line,
                section_number="unknown",
                hierarchy=["restored_content"],
                document_title=chunks[0].document_title if chunks else "unknown",
                parent_section="Restored Content",
                chapter_title="Restored Legal Content",
            )
            restored_chunks.append(new_chunk)

    logger.info("✅ Post-traitement terminé. Contenu juridique important restauré.")
    return restored_chunks
