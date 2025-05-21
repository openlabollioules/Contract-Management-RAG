import re
from typing import List, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import setup_logger

from .contract_splitter import Chunk, ContractSplitter

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class TextChunker:
    """
    Gère le chunking sémantique des textes juridiques, en respectant les structures hiérarchiques des contrats.

    Utilise le SemanticChunker de langchain_experimental pour regrouper les parties de texte
    sémantiquement proches, tout en préservant les frontières de sections importantes.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = 0.75,
        buffer_size: int = 8,
        number_of_chunks: Optional[int] = None,
        preserve_legal_structure: bool = True,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
    ):
        """
        Initialise le manager de chunking sémantique.

        Args:
            embedding_model_name: Nom du modèle d'embeddings à utiliser (all-mpnet-base-v2 ou bge-m3)
            breakpoint_threshold_type: Type de seuil pour déterminer les points de rupture ('percentile', 'standard_deviation', 'interquartile')
            breakpoint_threshold_amount: Valeur du seuil pour les points de rupture (selon le type)
            buffer_size: Taille du buffer entre chunks
            number_of_chunks: Nombre de chunks souhaité (optionnel)
            preserve_legal_structure: Si True, utilise des techniques spécifiques pour préserver la structure juridique
            chunk_size: Taille maximale des chunks générés
            chunk_overlap: Chevauchement entre chunks consécutifs
        """
        logger.info(
            f"Initialisation du TextChunker avec le modèle {embedding_model_name}"
        )
        logger.debug(
            f"Paramètres - breakpoint_threshold_type: {breakpoint_threshold_type}, buffer_size: {buffer_size}, number_of_chunks: {number_of_chunks}"
        )

        # Initialiser les embeddings HuggingFace pour le chunker
        logger.debug(
            f"Initialisation du modèle d'embeddings HuggingFace: {embedding_model_name}"
        )

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Stocker les paramètres de chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = self.embedding_model.embed_query
        logger.debug(
            f"Paramètres chunking - chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}"
        )

        # Initialiser le chunker sémantique avec les paramètres corrects
        logger.debug("Initialisation du chunker sémantique de langchain")
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embedding_model,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
        )
        
        # Initialiser également un chunker récursif pour le post-traitement
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )

        # Paramètres juridiques
        self.preserve_legal_structure = preserve_legal_structure
        logger.debug(f"Préservation de structure juridique: {preserve_legal_structure}")

        # Patterns juridiques avancés
        self.legal_patterns = [
            r"^(\d+(?:\.\d+)*)\s+[A-Z]",  # Format: "1.2 TITRE"
            r"^Article\s+\d+",  # Format: "Article 5"
            r"^Section\s+\d+",  # Format: "Section 3"
            r"^Clause\s+\d+",  # Format: "Clause 7"
            r"^ARTICLE\s+[IVX]+",  # Format romain: "ARTICLE IV"
            r"^[A-Z\s]{5,}:",  # Format: "DÉFINITIONS:"
            r"WHEREAS",  # Préambules
            r"NOW, THEREFORE",  # Clauses d'entrée
        ]
        logger.debug(f"Nombre de patterns juridiques: {len(self.legal_patterns)}")

        # Patterns de références croisées
        self.cross_ref_patterns = [
            r"pursuant to (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"as defined in (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"in accordance with (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"as per (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"according to (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"as set forth in (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"as specified in (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
            r"as stipulated in (?:Section|Article|Clause) (\d+(?:\.\d+)*)",
        ]
        logger.debug(
            f"Nombre de patterns de références croisées: {len(self.cross_ref_patterns)}"
        )
        logger.info("TextChunker initialization complete")

    def _preprocess_text_with_section_markers(self, text: str) -> str:
        """
        Prétraite le texte en ajoutant des marqueurs spéciaux aux débuts de sections importantes
        pour éviter que le semantic chunker ne coupe au milieu d'une unité logique.

        Args:
            text: Texte brut à prétraiter

        Returns:
            Texte prétraité avec marqueurs de section
        """
        logger.debug(
            f"Prétraitement du texte avec marqueurs de section (taille: {len(text)})"
        )
        # Ajouter des marqueurs spéciaux pour les patterns de section X., X.Y., X.Y.Z.
        # Ces marqueurs aideront le semantic chunker à respecter ces frontières
        lines = text.split("\n")
        processed_lines = []

        clause_starts = 0
        cross_refs = 0
        section_breaks = 0
        list_items = 0
        definitions = 0

        # Patterns pour les éléments de liste dans les contrats juridiques (améliorés)
        list_patterns = [
            # Éléments de liste numérotés ou alphabétiques (parenthèses ou non)
            r"^\s*(\([a-z]\)|\([ivx]+\)|[a-z]\)|[ivxlcdm]+\)|\([0-9]+\)|[0-9]+\)|\(\d+\.\d+\))",  # (a), (i), a), iv), (1), 1), (1.1)
            # Tirets et puces diverses
            r"^\s*(-|\*|\u2022|\u2023|\u25E6|\u2043|\u2219)\s",
            # Éléments de liste avec lettres suivies d'un point
            r"^\s*([A-Z]|[a-z]|[ivxlcdm]+|[IVXLCDM]+)\.(\s|$)",  # A., a., iv., IX.
            # Éléments introduits par une lettre ou un chiffre et un espace
            r"^\s*([a-z]|[A-Z]|[0-9])\s",  # a, b, c..., A, B, C..., 1, 2, 3...
            # Éléments du type 1.1, 1.2, etc.
            r"^\s*\d+\.\d+\s",
        ]
        
        # Patterns spécifiques pour les définitions
        definition_patterns = [
            r'^\s*".*"\s+is|means|shall mean',
            r'^\s*".*"\s+-\s+',
            r'^\s*[""].*[""] i(?:s|s\s+a|s\s+the|s\s+defined\s+as)',
            r'^\s*[Tt]he\s+term\s+[""].*[""]',
        ]

        # Marqueur de section précédente pour le contexte
        prev_section_level = None
        current_list_context = None

        for i, line in enumerate(lines):
            # Détecter les numéros de section et les structures juridiques principales
            if any(re.search(pattern, line) for pattern in self.legal_patterns):
                # Ajouter un marqueur spécial pour indiquer un début de clause ou section
                processed_lines.append("[CLAUSE_START]" + line)
                clause_starts += 1
                
                # Réinitialiser le contexte de liste
                current_list_context = None
                
                # Essayer de détecter le niveau de section
                section_match = re.search(r"^(\d+(?:\.\d+)*)", line)
                if section_match:
                    prev_section_level = len(section_match.group(1).split('.'))
                else:
                    prev_section_level = None
                    
            # Détecter les définitions et termes définis
            elif any(re.search(pattern, line, re.IGNORECASE) for pattern in definition_patterns):
                processed_lines.append("[DEFINITION]" + line)
                definitions += 1
                
            # Détecter les éléments de liste en tenant compte du contexte
            elif any(re.match(pattern, line) for pattern in list_patterns):
                # Déterminer le type de liste
                if re.match(r"^\s*\([a-z]\)|^\s*[a-z]\)", line):
                    list_type = "alpha"
                elif re.match(r"^\s*\([ivx]+\)|^\s*[ivxlcdm]+\)", line):
                    list_type = "roman"
                elif re.match(r"^\s*\([0-9]+\)|^\s*[0-9]+\)", line):
                    list_type = "numeric"
                elif re.match(r"^\s*(-|\*|\u2022)", line):
                    list_type = "bullet"
                else:
                    list_type = "generic"
                
                # Si c'est le début d'une nouvelle liste ou un changement de type
                if current_list_context != list_type:
                    processed_lines.append("[LIST_START:" + list_type + "]" + line)
                    current_list_context = list_type
                else:
                    # Continuation de la liste actuelle
                    processed_lines.append("[LIST_ITEM:" + list_type + "]" + line)
                
                list_items += 1
                
            # Détecter les références croisées pour enrichissement contextuel
            elif any(re.search(pattern, line) for pattern in self.cross_ref_patterns):
                # Marquer les références pour un traitement spécial
                processed_lines.append("[CROSS_REF]" + line)
                cross_refs += 1
                
            # Patterns génériques de section comme dans la version précédente
            elif any(
                re.match(pattern, line)
                for pattern in [
                    r"^\d+\.\s",  # Format X.
                    r"^\d+\.\d+\.\s",  # Format X.Y.
                    r"^\d+\.\d+\.\d+\.\s",  # Format X.Y.Z.
                ]
            ):
                # Ajouter un marqueur spécial pour indiquer un début de section important
                processed_lines.append("[SECTION_BREAK]" + line)
                section_breaks += 1
                
                # Mettre à jour le niveau de section
                section_match = re.match(r"^(\d+(?:\.\d+)*)", line)
                if section_match:
                    prev_section_level = len(section_match.group(1).split('.'))
                    
            # Détecter les phrases finales de paragraphes pour éviter la coupure
            elif line.strip().endswith(('.', ':', ';')) and i < len(lines) - 1 and lines[i+1].strip() and not lines[i+1].strip()[0].islower():
                processed_lines.append("[PARAGRAPH_END]" + line)
                
            # Détecter les phrases longues qui pourraient contenir des clauses importantes
            elif len(line.split()) > 20 and any(keyword in line.lower() for keyword in ["shall", "must", "will", "agree", "represent", "warrant", "indemnify", "liability"]):
                processed_lines.append("[IMPORTANT_CLAUSE]" + line)
                
            else:
                processed_lines.append(line)

        logger.info(
            f"Marqueurs ajoutés - Clauses: {clause_starts}, Éléments de liste: {list_items}, "
            f"Références croisées: {cross_refs}, Sections: {section_breaks}, Définitions: {definitions}"
        )
        return "\n".join(processed_lines)

    def _convert_to_chunks(
        self, semantic_chunks: List[str], document_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Convertit les chunks sémantiques en objets Chunk structurés avec métadonnées juridiques enrichies.

        Args:
            semantic_chunks: Liste de chunks textuels générés par le chunker sémantique
            document_title: Titre du document

        Returns:
            Liste d'objets Chunk avec métadonnées
        """
        logger.info(
            f"Conversion de {len(semantic_chunks)} chunks sémantiques en objets Chunk"
        )
        chunks = []

        for i, chunk_text in enumerate(semantic_chunks):
            # Supprimer les marqueurs de section qui ont été ajoutés
            cleaned_text = (
                chunk_text.replace("[CLAUSE_START]", "")
                .replace("[SECTION_BREAK]", "")
                .replace("[CROSS_REF]", "")
                .replace("[LIST_ITEM]", "")
            )

            # Détecter la première ligne pour essayer d'extraire un numéro de section
            lines = cleaned_text.split("\n")
            section_number = None
            clause_type = self._detect_clause_type(cleaned_text)

            # Chercher un numéro de section dans les premières lignes
            for line in lines[:5]:  # Regarder les 5 premières lignes
                # Pattern simple pour la détection de numéro de section (peut être amélioré)
                import re

                match = re.search(r"^(\d+(?:\.\d+)*)", line)
                if match:
                    section_number = match.group(1)
                    logger.debug(
                        f"Numéro de section détecté dans le chunk {i}: {section_number}"
                    )
                    break

            # Créer un objet Chunk avec métadonnées juridiques enrichies
            chunk = Chunk(
                content=cleaned_text,
                section_number=section_number,
                document_title=document_title,
                hierarchy=(
                    self._extract_hierarchy(section_number) if section_number else None
                ),
                parent_section=(
                    self._extract_parent_section(section_number)
                    if section_number
                    else None
                ),
                chapter_title=None,  # Pourrait être extrait si nécessaire
            )

            # Ajouter des métadonnées spécifiques au domaine juridique
            if hasattr(chunk, "metadata"):
                chunk.metadata = {}
            else:
                setattr(chunk, "metadata", {})

            chunk.metadata["clause_type"] = clause_type
            chunk.metadata["has_cross_references"] = "[CROSS_REF]" in chunk_text
            chunk.metadata["chunk_index"] = i

            logger.debug(
                f"Chunk {i} créé - Section: {section_number}, Type: {clause_type}, Taille: {len(cleaned_text)}"
            )
            chunks.append(chunk)

        logger.info(f"Conversion terminée: {len(chunks)} objets Chunk créés")
        return chunks

    def _detect_clause_type(self, text: str) -> str:
        """
        Détecte le type de clause juridique dans un chunk de texte.

        Args:
            text: Texte du chunk à analyser

        Returns:
            Type de clause détecté
        """
        # Dictionnaire des patterns pour différents types de clauses
        clause_patterns = {
            "confidentialité": r"\b(confidential|confidentialité|non[\s-]disclosure|secret|divulgation)\b",
            "résiliation": r"\b(termin|résili|cancel|end.*agreement|fin.*contrat)\w*\b",
            "force majeure": r"\b(force\s+majeure|act\s+of\s+god|événement.*extérieur)\b",
            "indemnisation": r"\b(indemn|compensat|dommages|damages)\w*\b",
            "limitation de responsabilité": r"\b(limit\w*\s+(de\s+)?(responsabilit|liabilit)|liabilit\w*\s+limit)\b",
            "propriété intellectuelle": r"\b(intellect\w*\s+propert|propr\w*\s+intellect|ip\s+rights|brevet|patent|copyright|droit.*auteur)\b",
            "garanties": r"\b(warrant|garantie|represent\w*)\b",
            "paiement": r"\b(pay\w*|paiement|compensat\w*|prix|price|fee|frais)\b",
            "durée": r"\b(term|durée|period|période)\b",
            "résolution des litiges": r"\b(disput|litige|arbitr\w*|médiati\w*|juridic\w*|tribuna\w*|court)\b",
            "définitions": r"\b(défini\w+|signifi\w+|mean\w+|terme\w*)\b",
            "cession": r"\b(assign\w*|cessi\w*|transf[ée]r\w*)\b",
            "non-concurrence": r"\b(non[\s-]comp[eé]t|concurren\w+)\b",
            "modification": r"\b(modifi\w*|amend\w*|chang\w*)\b",
        }

        # Recherche du type de clause basé sur les patterns définis
        for clause_type, pattern in clause_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return clause_type

        # Analyse des titres de section pour détecter le type de clause
        first_lines = text.split("\n")[:3]  # Analyser les premières lignes
        for line in first_lines:
            line = line.strip().lower()
            for clause_type in clause_patterns.keys():
                if clause_type in line:
                    return clause_type

        # Par défaut, si aucun type spécifique n'est identifié
        return "autre"

    def _extract_hierarchy(self, section_number: str) -> List[str]:
        """
        Extrait la hiérarchie à partir d'un numéro de section.
        Par exemple, pour "3.4.1", retourne ["3", "3.4", "3.4.1"]
        Ou pour les formats comme (a), (i), etc., construit une hiérarchie appropriée.

        Args:
            section_number: Numéro de section (ex: "3.4.1") ou identifiant de liste (ex: "(a)", "(i)")

        Returns:
            Liste de numéros de section représentant la hiérarchie
        """
        if not section_number:
            return []
            
        # Cas spécial pour les éléments de liste comme (a), (i), a), i), etc.
        list_item_patterns = {
            'alphabetic': r'^\(?([a-z])\)?$',             # (a), a)
            'roman': r'^\(?([ivxlcdm]+)\)?$',             # (i), i), (iv), iv)
            'numeric': r'^\(?(\d+)\)?$',                  # (1), 1)
            'alphanumeric': r'^\(?(\d+\.\d+[a-z]*)\)?$'   # (1.2a), 1.2a)
        }
        
        for list_type, pattern in list_item_patterns.items():
            match = re.match(pattern, section_number, re.IGNORECASE)
            if match:
                item_value = match.group(1)
                # Pour les listes, nous retournons simplement l'élément lui-même car nous n'avons pas
                # d'informations sur la hiérarchie supérieure dans ce contexte
                return [section_number]

        # Cas standard pour les hiérarchies numériques (X.Y.Z)
        parts = section_number.split(".")
        hierarchy = []
        current = ""

        for part in parts:
            if current:
                current += f".{part}"
            else:
                current = part
            hierarchy.append(current)

        return hierarchy

    def _extract_parent_section(self, section_number: str) -> Optional[str]:
        """
        Extrait le numéro de section parent.
        Par exemple, pour "3.4.1", retourne "3.4"

        Args:
            section_number: Numéro de section (ex: "3.4.1")

        Returns:
            Numéro de la section parente ou None
        """
        if not section_number or "." not in section_number:
            return None

        parts = section_number.split(".")
        return ".".join(parts[:-1])

    def _calculate_optimal_threshold(self, text: str) -> float:
        """
        Calcule le seuil optimal pour le chunking sémantique basé sur la complexité du texte.
        Optimisé pour all-mpnet-base-v2 et bge-m3.

        Args:
            text: Texte à analyser

        Returns:
            Seuil optimal
        """
        # Analyse de base de la complexité du texte
        avg_sentence_length = np.mean(
            [len(s.split()) for s in re.split(r"[.!?]", text) if s]
        )
        legal_terms_count = len(
            re.findall(
                r"\b(pursuant|hereto|thereof|herein|aforesaid|whereof|hereunder)\b",
                text,
                re.IGNORECASE,
            )
        )

        # Détection de sections juridiques complexes
        complex_sections = len(
            re.findall(
                r"\b(indemnification|representations|warranties|termination|confidentiality|assignment)\b",
                text,
                re.IGNORECASE,
            )
        )

        # Détection des références croisées, qui indiquent une forte interconnexion des clauses
        cross_references = len(
            re.findall(r"(Section|Article|Paragraph|Clause)\s+\d+(\.\d+)*", text)
        )

        # Détection de la densité de termes définis (généralement en majuscules dans les contrats)
        defined_terms = len(re.findall(r"\b[A-Z]{2,}[A-Za-z]*\b", text))

        # Analyse de la structure - nombres de niveaux de titres
        section_depth = 0
        for line in text.split("\n"):
            if re.match(r"^[0-9]+\.[0-9]+\.[0-9]+\.", line):
                section_depth = max(section_depth, 3)
            elif re.match(r"^[0-9]+\.[0-9]+\.", line):
                section_depth = max(section_depth, 2)
            elif re.match(r"^[0-9]+\.", line):
                section_depth = max(section_depth, 1)

        # Calcul du seuil basé sur les indicateurs de complexité et le modèle d'embedding
        # Valeur par défaut optimisée pour all-mpnet-base-v2 et bge-m3
        base_threshold = 0.65

        # Ajustement pour la longueur des phrases
        if avg_sentence_length > 35:
            base_threshold -= 0.05  # Phrases plus longues = seuil plus bas pour plus de découpage
        elif avg_sentence_length < 15:
            base_threshold += 0.05  # Phrases courtes = moins de découpage nécessaire

        # Ajustement pour les termes juridiques et clauses complexes
        complexity_factor = (
            (legal_terms_count + complex_sections * 2) / max(len(text.split()), 1) * 100
        )
        if complexity_factor > 5:
            base_threshold -= 0.1  # Texte juridique complexe = plus de découpage

        # Ajustement pour les références croisées
        if cross_references > 10:
            base_threshold -= 0.05  # Beaucoup de références = plus de découpage nécessaire

        # Ajustement pour les termes définis
        defined_terms_density = defined_terms / max(len(text.split()), 1) * 100
        if defined_terms_density > 8:
            base_threshold -= 0.05  # Beaucoup de termes définis = texte plus technique

        # Ajustement pour la profondeur des sections
        if section_depth >= 3:
            base_threshold -= 0.05  # Structure complexe = plus de découpage

        # Limites pour éviter des valeurs extrêmes
        return max(0.4, min(0.8, base_threshold))

    def _detect_dates(self, text: str) -> List[str]:
        """
        Detect dates in text using regex patterns.
        Supports various date formats commonly found in contracts.

        Args:
            text: Text to analyze

        Returns:
            List of detected dates
        """
        print("detect_dates")
        # Common date patterns in contracts
        date_patterns = [
            # DD/MM/YYYY or DD-MM-YYYY
            r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b',
            # YYYY/MM/DD or YYYY-MM-DD
            r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',
            # Month DD, YYYY (e.g., "January 1, 2024")
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01]),\s+(?:19|20)\d{2}\b',
            # DD Month YYYY (e.g., "1 January 2024")
            r'\b(?:0?[1-9]|[12][0-9]|3[01])\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',
            # French date format (e.g., "le 1er janvier 2024")
            r'\ble\s+(?:0?[1-9]|[12][0-9]|3[01])(?:er|ème)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(?:19|20)\d{2}\b',
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                print(f"Date trouvée: {match.group(0)}")
                dates.append(match.group(0))

        return dates

    def chunk_text(
        self, text: str, doc_id: str = None, doc_metadata: dict = None
    ) -> List[Chunk]:
        """
        Divise un texte en chunks sémantiques en préservant la structure du document.

        Args:
            text: Texte à diviser en chunks
            doc_id: Identifiant du document (optionnel)
            doc_metadata: Métadonnées du document (titre, date, etc.)

        Returns:
            Liste de chunks avec métadonnées enrichies
        """
        if not text:
            return []

        # Préserver les métadonnées du document source
        doc_metadata = doc_metadata or {}
        doc_title = doc_metadata.get("title", "Document sans titre")

        logger.info(f"Démarrage du découpage du document: {doc_title} ({len(text)} caractères)")
        
        # Prétraiter le texte pour marquer les sections importantes
        if self.preserve_legal_structure:
            processed_text = self._preprocess_text_with_section_markers(text)
            logger.debug(f"Texte prétraité avec marqueurs de section")
        else:
            processed_text = text
            
        # Division initiale par grandes sections pour améliorer le découpage sémantique
        logger.info("Division préliminaire par grandes sections...")
        # Utiliser une expression régulière pour détecter les sections principales (titres de niveau 1 et 2)
        main_section_breaks = re.split(r'((?:^|\n)#{1,2}\s+.*?)(?=(?:^|\n)#{1,2}\s+|\Z)|(?:^|\n)(?:\d+\.(?:\s|\n)+[A-Z].*?)(?=(?:^|\n)\d+\.(?:\s|\n)+[A-Z]|\Z)|(?:^|\n)(?:[A-Z][A-Z\s]+:)(?=(?:^|\n)[A-Z][A-Z\s]+:|\Z)', processed_text)
        
        main_sections = []
        for section in main_section_breaks:
            if section and section.strip():
                main_sections.append(section)
        
        logger.info(f"Document divisé en {len(main_sections)} grandes sections")
            
        # Traiter chaque grande section séparément pour un meilleur découpage sémantique
        all_chunks_text = []
        
        for section_idx, section_text in enumerate(main_sections):
            logger.debug(f"Traitement de la section {section_idx+1}/{len(main_sections)} (taille: {len(section_text)})")
            
            # Si la section est suffisamment petite et inférieure à la taille maximale de chunk, la garder intacte
            if len(section_text.split()) <= self.chunk_size * 0.8:
                all_chunks_text.append(section_text)
                continue
                
            # Utiliser le semantic chunker pour cette section
            try:
                section_chunks = self.semantic_chunker.split_text(section_text)
                # Vérifier que le semantic chunker a bien divisé la section
                if section_chunks and len(section_chunks) > 1:
                    all_chunks_text.extend(section_chunks)
                else:
                    # Fallback au découpage récursif pour cette section
                    fallback_chunks = self.recursive_chunker.split_text(section_text)
                    all_chunks_text.extend(fallback_chunks)
            except Exception as e:
                logger.error(f"Erreur lors du découpage sémantique de la section {section_idx+1}: {e}")
                # Fallback au découpage récursif
                fallback_chunks = self.recursive_chunker.split_text(section_text)
                all_chunks_text.extend(fallback_chunks)
        
        logger.info(f"Découpage initial: {len(all_chunks_text)} chunks générés")
        
        # Optimisation supplémentaire: fusion des chunks trop petits et préservation des listes
        processed_chunks_text = []
        
        # Définir les patterns pour détecter les éléments de liste complets
        list_patterns = {
            'alpha': r"^\s*(?:\([a-z]\)|[a-z]\))",
            'roman': r"^\s*(?:\([ivx]+\)|[ivxlcdm]+\))",
            'numeric': r"^\s*(?:\([0-9]+\)|[0-9]+\))",
            'bullet': r"^\s*(?:-|\*|\u2022)",
        }
        
        # Fonction pour détecter le type de liste dans un chunk
        def detect_list_type(chunk_text):
            list_types = []
            lines = chunk_text.split('\n')
            
            for line in lines:
                if re.search(r"\[LIST_START:(\w+)\]", line):
                    list_type = re.search(r"\[LIST_START:(\w+)\]", line).group(1)
                    if list_type not in list_types:
                        list_types.append(list_type)
                elif re.search(r"\[LIST_ITEM:(\w+)\]", line):
                    list_type = re.search(r"\[LIST_ITEM:(\w+)\]", line).group(1)
                    if list_type not in list_types:
                        list_types.append(list_type)
            
            return list_types
        
        # Fonction pour vérifier si un chunk contient des éléments importants
        def is_important_chunk(chunk_text):
            # Vérifier si le chunk contient des définitions importantes
            if "[DEFINITION]" in chunk_text:
                return True
                
            # Vérifier si le chunk contient des clauses importantes
            if "[IMPORTANT_CLAUSE]" in chunk_text:
                return True
                
            # Vérifier le contenu du chunk pour des mots-clés juridiques importants
            important_keywords = ["shall", "must", "will not", "shall not", "represent", "warrant", 
                                 "liability", "indemnify", "termination", "dispute", "governing law",
                                 "force majeure", "confidential", "warranty"]
            
            for keyword in important_keywords:
                if keyword in chunk_text.lower():
                    return True
                    
            return False
        
        # Combiner les chunks de manière optimale
        i = 0
        while i < len(all_chunks_text):
            current_chunk = all_chunks_text[i]
            current_size = len(current_chunk.split())
            
            # 1. Si le chunk actuel est trop petit (moins de 25% de la taille cible)
            if current_size < self.chunk_size * 0.25:
                # Vérifier si c'est un chunk important malgré sa petite taille
                if is_important_chunk(current_chunk):
                    # Garder les chunks importants, même s'ils sont petits
                    processed_chunks_text.append(current_chunk)
                    i += 1
                    continue
                
                # Si ce n'est pas le dernier chunk, essayer de fusionner avec le suivant
                if i < len(all_chunks_text) - 1:
                    next_chunk = all_chunks_text[i+1]
                    next_size = len(next_chunk.split())
                    
                    # Si la fusion reste en dessous de la taille cible
                    if current_size + next_size <= self.chunk_size:
                        all_chunks_text[i+1] = current_chunk + "\n\n" + next_chunk
                        i += 1
                        continue
                
                # Si c'est le dernier chunk ou si la fusion dépasserait la taille cible
                processed_chunks_text.append(current_chunk)
                
            # 2. Si le chunk est de taille moyenne (entre 25% et 75% de la taille cible)
            elif current_size < self.chunk_size * 0.75:
                # Vérifier si ce chunk démarre une liste
                current_list_types = detect_list_type(current_chunk)
                
                # Si le chunk démarre une liste et qu'il y a un chunk suivant
                if current_list_types and i < len(all_chunks_text) - 1:
                    next_chunk = all_chunks_text[i+1]
                    next_size = len(next_chunk.split())
                    next_list_types = detect_list_type(next_chunk)
                    
                    # Si le prochain chunk continue avec le même type de liste
                    if any(lt in next_list_types for lt in current_list_types) and current_size + next_size <= self.chunk_size:
                        # Fusionner pour préserver la liste complète
                        all_chunks_text[i+1] = current_chunk + "\n\n" + next_chunk
                        i += 1
                        continue
                
                processed_chunks_text.append(current_chunk)
                
            # 3. Si le chunk est de bonne taille
            else:
                processed_chunks_text.append(current_chunk)
            
            i += 1
        
        logger.info(f"Après fusion des petits chunks et préservation des listes: {len(processed_chunks_text)} chunks finaux")
        
        # Nettoyer les marqueurs spéciaux dans tous les chunks
        clean_chunks_text = []
        for chunk_text in processed_chunks_text:
            # Supprimer tous les marqueurs ajoutés lors de la préparation
            clean_text = (chunk_text
                .replace("[CLAUSE_START]", "")
                .replace("[SECTION_BREAK]", "")
                .replace("[CROSS_REF]", "")
                .replace("[PARAGRAPH_END]", "")
                .replace("[IMPORTANT_CLAUSE]", "")
                .replace("[DEFINITION]", ""))
                
            # Supprimer les marqueurs de liste avec leur type
            clean_text = re.sub(r"\[LIST_START:\w+\]", "", clean_text)
            clean_text = re.sub(r"\[LIST_ITEM:\w+\]", "", clean_text)
            
            clean_chunks_text.append(clean_text)
            
        # Construction des chunks enrichis avec métadonnées
        chunks = []
        for i, chunk_text in enumerate(clean_chunks_text):
            # Chercher un numéro de section dans les premières lignes
            section_match = None
            for pattern in [
                r"[-\*]\s*(\d+(?:\.\d+)*)",  # Format: - 1.2
                r"^(\d+(?:\.\d+)*)\s",       # Format: 1.2 au début de ligne
                r"Article\s+(\d+(?:\.\d+)*)", # Format: Article 5
                r"Section\s+(\d+(?:\.\d+)*)", # Format: Section 3
                r"Clause\s+(\d+(?:\.\d+)*)",  # Format: Clause 7
                r"^\s*(\([a-z]\)|\([ivx]+\)|[a-z]\)|[ivxlcdm]+\))", # Format: (a), (i), a), iv)
            ]:
                match = re.search(pattern, chunk_text, re.MULTILINE)
                if match:
                    section_match = match
                    break
            
            section_number = section_match.group(1) if section_match else None
            
            # Déterminer un titre de chapitre
            chapter_title = None
            lines = chunk_text.split("\n")
            for line in lines[:5]:  # Regarder les 5 premières lignes
                line = line.strip()
                if line and 10 < len(line) < 100:
                    # Rechercher un titre probable (motifs courants dans les contrats)
                    if (line.isupper() or
                        re.match(r"^\d+\.\s+[A-Z]", line) or
                        re.match(r"^[A-Z][a-z]+\s+\d+", line) or
                        re.match(r"^[-\*]\s*\d+\.\d+\s+[A-Z]", line)):
                        chapter_title = line
                        break
            
            # Si aucun titre n'a été trouvé, utiliser la première ligne non vide
            if not chapter_title:
                for line in lines:
                    if line.strip():
                        chapter_title = line.strip()[:100]
                        break
            
            # Création du chunk avec toutes les métadonnées
            chunk = Chunk(
                content=chunk_text,
                section_number=section_number,
                document_title=doc_title,
                hierarchy=self._extract_hierarchy(section_number) if section_number else None,
                parent_section=self._extract_parent_section(section_number) if section_number else None,
                chapter_title=chapter_title,
            )
            
            # Ajouter la position relative dans le document
            setattr(chunk, "position", i + 1)
            setattr(chunk, "total_chunks", len(clean_chunks_text))
            
            chunks.append(chunk)
        
        logger.info(f"Création terminée: {len(chunks)} chunks avec métadonnées enrichies")
        return chunks
