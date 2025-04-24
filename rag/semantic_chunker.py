import re
from typing import List, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from utils.logger import setup_logger

from .intelligent_splitter import Chunk, IntelligentSplitter

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class SemanticChunkManager:
    """
    Gère le chunking sémantique des textes juridiques, en respectant les structures hiérarchiques des contrats.

    Utilise le SemanticChunker de langchain_experimental pour regrouper les parties de texte
    sémantiquement proches, tout en préservant les frontières de sections importantes.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        buffer_size: int = 3,
        number_of_chunks: Optional[int] = None,
        preserve_legal_structure: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialise le manager de chunking sémantique.

        Args:
            embedding_model_name: Nom du modèle d'embeddings à utiliser
            breakpoint_threshold_type: Type de seuil pour déterminer les points de rupture ('percentile', 'standard_deviation', 'interquartile')
            breakpoint_threshold_amount: Valeur du seuil pour les points de rupture (selon le type)
            buffer_size: Taille du buffer entre chunks
            number_of_chunks: Nombre de chunks souhaité (optionnel)
            preserve_legal_structure: Si True, utilise des techniques spécifiques pour préserver la structure juridique
            chunk_size: Taille maximale des chunks générés
            chunk_overlap: Chevauchement entre chunks consécutifs
        """
        logger.info(
            f"Initialisation du SemanticChunkManager avec le modèle {embedding_model_name}"
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
        ]
        logger.debug(
            f"Nombre de patterns de références croisées: {len(self.cross_ref_patterns)}"
        )
        logger.info("SemanticChunkManager initialization complete")

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

        for line in lines:
            # Détecter les numéros de section comme X., X.Y., X.Y.Z. et les structures juridiques
            if any(re.search(pattern, line) for pattern in self.legal_patterns):
                # Ajouter un marqueur spécial pour indiquer un début de clause ou section
                processed_lines.append("[CLAUSE_START]" + line)
                clause_starts += 1
            # Détecter les références croisées pour enrichissement contextuel
            elif any(re.search(pattern, line) for pattern in self.cross_ref_patterns):
                # Marquer les références pour un traitement spécial
                processed_lines.append("[CROSS_REF]" + line)
                cross_refs += 1
            # Patterns génériques de section comme dans la version précédente
            elif any(
                pattern in line
                for pattern in [
                    r"^\d+\.\s",  # Format X.
                    r"^\d+\.\d+\.\s",  # Format X.Y.
                    r"^\d+\.\d+\.\d+\.\s",  # Format X.Y.Z.
                ]
            ):
                # Ajouter un marqueur spécial pour indiquer un début de section important
                processed_lines.append("[SECTION_BREAK]" + line)
                section_breaks += 1
            else:
                processed_lines.append(line)

        logger.info(
            f"Marqueurs ajoutés - Clauses: {clause_starts}, Références croisées: {cross_refs}, Sections: {section_breaks}"
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

        Args:
            section_number: Numéro de section (ex: "3.4.1")

        Returns:
            Liste de numéros de section représentant la hiérarchie
        """
        if not section_number:
            return []

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

        # Calcul du seuil basé sur les indicateurs de complexité
        base_threshold = 0.5  # Valeur par défaut

        # Ajustement pour la longueur des phrases
        if avg_sentence_length > 35:
            base_threshold -= (
                0.1  # Phrases plus longues = seuil plus bas pour plus de découpage
            )
        elif avg_sentence_length < 15:
            base_threshold += 0.05  # Phrases courtes = moins de découpage nécessaire

        # Ajustement pour les termes juridiques et clauses complexes
        complexity_factor = (
            (legal_terms_count + complex_sections * 2) / max(len(text.split()), 1) * 100
        )
        if complexity_factor > 5:
            base_threshold -= 0.15  # Texte juridique complexe = plus de découpage

        # Ajustement pour les références croisées
        if cross_references > 10:
            base_threshold -= (
                0.1  # Beaucoup de références = plus de découpage nécessaire
            )

        # Ajustement pour les termes définis
        defined_terms_density = defined_terms / max(len(text.split()), 1) * 100
        if defined_terms_density > 8:
            base_threshold -= 0.05  # Beaucoup de termes définis = texte plus technique

        # Ajustement pour la profondeur des sections
        if section_depth >= 3:
            base_threshold -= 0.1  # Structure complexe = plus de découpage

        # Limites pour éviter des valeurs extrêmes
        return max(0.2, min(0.8, base_threshold))

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

        # Détection des sections principales du document
        section_pattern = r"^(?:\d+\.)+\s*(.*?)$|^(?:[A-Z][A-Z\s]+)[:.]"
        section_headers = []
        current_position = 0
        for match in re.finditer(section_pattern, text, re.MULTILINE):
            section_headers.append(
                {
                    "title": match.group(0).strip(),
                    "position": match.start(),
                    "level": (
                        len(match.group(0).split(".")) - 1
                        if match.group(0).count(".") > 0
                        else 0
                    ),
                }
            )

        # Extraction des termes définis avec regex
        defined_terms = {}
        term_pattern = (
            r'"([^"]+)"\s+(?:signifie|désigne|means|shall\s+mean|refers\s+to)'
        )
        for match in re.finditer(term_pattern, text, re.IGNORECASE):
            term = match.group(1)
            defined_terms[term] = True

        # Extraction des références aux articles avec regex
        references = {}
        ref_pattern = r"(?:l\'article|section|clause|article)\s+(\d+(?:\.\d+)*)"
        for match in re.finditer(ref_pattern, text, re.IGNORECASE):
            reference = match.group(1)
            references[reference] = True

        # Utilisation du SemanticChunker pour la segmentation sémantique
        # Utiliser le chunker déjà configuré dans l'initialisation
        semantic_chunks = self.semantic_chunker.create_documents([text])[
            0
        ].page_content.split("\n\n")

        # Construction des chunks enrichis avec métadonnées
        chunks = []
        for i, chunk_text in enumerate(semantic_chunks):
            if not chunk_text.strip():  # Ignorer les chunks vides
                continue

            chunk_start = text.find(chunk_text)
            if chunk_start == -1:  # Gestion des cas où le texte exact n'est pas trouvé
                continue

            # Détermination de la section courante pour ce chunk
            current_section = None
            section_hierarchy = []
            for j in range(len(section_headers) - 1, -1, -1):
                if section_headers[j]["position"] <= chunk_start:
                    current_section = section_headers[j]["title"]
                    # Construction de la hiérarchie des sections parentes
                    current_level = section_headers[j]["level"]
                    section_hierarchy = [current_section]
                    for k in range(j - 1, -1, -1):
                        if section_headers[k]["level"] < current_level:
                            section_hierarchy.insert(0, section_headers[k]["title"])
                            current_level = section_headers[k]["level"]
                    break

            # Détection du type de clause pour ce chunk
            clause_type = self._detect_clause_type(chunk_text)

            # Extraction des termes définis dans ce chunk spécifique
            chunk_defined_terms = []
            for term in defined_terms:
                if term in chunk_text:
                    chunk_defined_terms.append(term)

            # Extraction des références dans ce chunk spécifique
            chunk_references = []
            for ref in references:
                if ref in chunk_text:
                    chunk_references.append(ref)

            # Création d'un identifiant unique pour le chunk
            chunk_id = f"{doc_id or 'doc'}_{i}"

            # Construction du chunk final
            chunk = Chunk(
                content=chunk_text,
                section_number=section_hierarchy[0] if section_hierarchy else None,
                document_title=doc_title,
                hierarchy=section_hierarchy,
                parent_section=(
                    section_hierarchy[-2] if len(section_hierarchy) > 1 else None
                ),
                chapter_title=current_section,
            )

            # Ajouter des métadonnées spécifiques en tant qu'attributs
            setattr(chunk, "clause_type", clause_type)
            setattr(chunk, "has_cross_references", "[CROSS_REF]" in chunk_text)
            setattr(chunk, "chunk_index", i)
            setattr(chunk, "doc_id", doc_id)
            setattr(chunk, "chunk_id", chunk_id)
            setattr(chunk, "defined_terms", chunk_defined_terms[:10])
            setattr(chunk, "references", chunk_references[:10])
            setattr(chunk, "position", i)
            setattr(chunk, "total_chunks", len(semantic_chunks))

            # Ajouter les métadonnées du document original en tant qu'attributs
            for key, value in doc_metadata.items():
                setattr(chunk, key, value)

            chunks.append(chunk)

        return chunks

    def hybrid_chunk_text(
        self, text: str, document_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Implémente un chunking hybride qui combine le découpage structurel et sémantique.

        Args:
            text: Texte à découper
            document_title: Titre du document

        Returns:
            Liste de chunks avec métadonnées enrichies
        """
        # 1. Découpage structurel initial
        structure_splitter = IntelligentSplitter(document_title=document_title)
        initial_chunks = structure_splitter.split(text)

        final_chunks = []

        # 2. Traitement de chaque chunk structurel
        for chunk in initial_chunks:
            # Si le chunk est petit, le garder tel quel
            if len(chunk.content.split()) <= 800:  # ~800 tokens
                final_chunks.append(chunk)
            else:
                # Pour les sections longues, appliquer le chunking sémantique
                # Préserver les métadonnées de la section
                section_metadata = {
                    "section_number": chunk.section_number,
                    "hierarchy": chunk.hierarchy,
                    "document_title": chunk.document_title,
                    "parent_section": chunk.parent_section,
                    "chapter_title": chunk.chapter_title,
                }

                # Appliquer le chunking sémantique
                semantic_chunks = self.semantic_chunker.create_documents(
                    [chunk.content]
                )

                # Convertir en objets Chunk avec métadonnées préservées
                for semantic_chunk in semantic_chunks:
                    new_chunk = Chunk(
                        content=semantic_chunk.page_content, **section_metadata
                    )
                    final_chunks.append(new_chunk)

        return final_chunks
