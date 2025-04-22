from typing import List, Optional

from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import HuggingFaceEmbeddings

from .intelligent_splitter import Chunk


class SemanticChunkManager:
    """
    Gère le chunking sémantique des textes, en respectant les structures hiérarchiques des contrats juridiques.
    
    Utilise le SemanticChunker de langchain_experimental pour regrouper les parties de texte
    sémantiquement proches, tout en préservant les frontières de sections importantes.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_threshold: float = 0.25,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        """
        Initialise le manager de chunking sémantique.
        
        Args:
            embedding_model_name: Nom du modèle d'embeddings à utiliser
            breakpoint_threshold: Seuil de dissimilarité pour créer une rupture entre chunks (plus élevé = moins de ruptures)
            min_chunk_size: Taille minimale d'un chunk en caractères
            max_chunk_size: Taille maximale d'un chunk en caractères
        """
        # Initialiser les embeddings HuggingFace pour le chunker
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialiser le chunker sémantique
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold=breakpoint_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        
    def _preprocess_text_with_section_markers(self, text: str) -> str:
        """
        Prétraite le texte en ajoutant des marqueurs spéciaux aux débuts de sections importantes
        pour éviter que le semantic chunker ne coupe au milieu d'une unité logique.
        
        Args:
            text: Texte brut à prétraiter
            
        Returns:
            Texte prétraité avec marqueurs de section
        """
        # Ajouter des marqueurs spéciaux pour les patterns de section X., X.Y., X.Y.Z.
        # Ces marqueurs aideront le semantic chunker à respecter ces frontières
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Détecter les numéros de section comme X., X.Y., X.Y.Z.
            if any(pattern in line for pattern in [
                # Patterns pour détecter les numéros de section
                # Format X.
                r'^\d+\.\s',
                # Format X.Y.
                r'^\d+\.\d+\.\s',
                # Format X.Y.Z.
                r'^\d+\.\d+\.\d+\.\s',
            ]):
                # Ajouter un marqueur spécial pour indiquer un début de section important
                processed_lines.append(f"[SECTION_BREAK]{line}")
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _convert_to_chunks(self, semantic_chunks: List[str], document_title: Optional[str] = None) -> List[Chunk]:
        """
        Convertit les chunks sémantiques en objets Chunk structurés.
        
        Args:
            semantic_chunks: Liste de chunks textuels générés par le chunker sémantique
            document_title: Titre du document
            
        Returns:
            Liste d'objets Chunk avec métadonnées
        """
        chunks = []
        
        for i, chunk_text in enumerate(semantic_chunks):
            # Supprimer les marqueurs de section qui ont été ajoutés
            cleaned_text = chunk_text.replace("[SECTION_BREAK]", "")
            
            # Détecter la première ligne pour essayer d'extraire un numéro de section
            lines = cleaned_text.split('\n')
            section_number = None
            
            # Chercher un numéro de section dans les premières lignes
            for line in lines[:5]:  # Regarder les 5 premières lignes
                # Pattern simple pour la détection de numéro de section (peut être amélioré)
                import re
                match = re.search(r'^(\d+(?:\.\d+)*)', line)
                if match:
                    section_number = match.group(1)
                    break
            
            # Créer un objet Chunk
            chunk = Chunk(
                content=cleaned_text,
                section_number=section_number,
                document_title=document_title,
                hierarchy=self._extract_hierarchy(section_number) if section_number else None,
                parent_section=self._extract_parent_section(section_number) if section_number else None,
                chapter_title=None  # Pourrait être extrait si nécessaire
            )
            
            chunks.append(chunk)
        
        return chunks
    
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
            
        parts = section_number.split('.')
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
        if not section_number or '.' not in section_number:
            return None
            
        parts = section_number.split('.')
        return '.'.join(parts[:-1])
    
    def chunk_text(self, text: str, document_title: Optional[str] = None) -> List[Chunk]:
        """
        Découpe le texte en chunks sémantiques intelligents, en préservant la structure du document.
        
        Args:
            text: Texte à découper
            document_title: Titre du document (optionnel)
            
        Returns:
            Liste d'objets Chunk avec métadonnées
        """
        print("\n🔍 Découpage sémantique du texte...")
        
        # Prétraitement pour ajouter des marqueurs de section
        processed_text = self._preprocess_text_with_section_markers(text)
        
        # Utiliser le semantic chunker pour découper le texte
        semantic_chunks = self.semantic_chunker.split_text(processed_text)
        
        print(f"📊 Chunks sémantiques créés: {len(semantic_chunks)}")
        
        # Convertir en objets Chunk
        chunks = self._convert_to_chunks(semantic_chunks, document_title)
        
        return chunks 