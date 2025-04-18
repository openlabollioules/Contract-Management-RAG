import platform
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

# Détection de l'architecture
is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
if is_apple_silicon:
    print("🍎 Détection d'un processeur Apple Silicon")
    if torch.backends.mps.is_available():
        print("🎮 GPU MPS disponible")
        device = torch.device("mps")
    else:
        print("⚠️ GPU MPS non disponible, utilisation du CPU")
        device = torch.device("cpu")
else:
    print("💻 Architecture non Apple Silicon")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Utilisation du device: {device}")


@dataclass
class Chunk:
    content: str
    section_number: Optional[str] = None
    document_title: Optional[str] = None
    hierarchy: Optional[List[str]] = None
    parent_section: Optional[str] = None  # Numéro de la section parente
    chapter_title: Optional[str] = None  # Titre du chapitre


class IntelligentSplitter:
    def __init__(self, document_title: Optional[str] = None):
        self.document_title = document_title
        self.chunk_count = 0
        self.section_hierarchy = {}  # Pour stocker la hiérarchie des sections
        self.section_titles = {}  # Stocke les titres des chapitres (ex: "3": "Work")

    def _normalize_section_number(self, section_number: str) -> str:
        """Normalise un numéro de section en ajoutant un point entre les chiffres séparés par un espace."""
        if not section_number:
            return section_number

        # Nettoyer le numéro de section
        section_number = section_number.strip()

        # Si le numéro contient des espaces, les remplacer par des points
        if " " in section_number:
            # Remplacer les espaces multiples par un seul point
            normalized = re.sub(r"\s+", ".", section_number)
            # S'assurer qu'il n'y a pas de points multiples
            normalized = re.sub(r"\.+", ".", normalized)
            # Supprimer les points en début et fin
            normalized = normalized.strip(".")
            return normalized

        return section_number

    def _is_chapter_title(self, line: str) -> Optional[Tuple[str, str]]:
        """Détecte si une ligne est un titre de chapitre (ex: '3. Work')."""
        pattern = r"^(\d+)\.\s+(.*?)$"
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1), match.group(2).strip()
        return None

    def _is_subsection(self, line: str) -> Optional[str]:
        """Détecte si une ligne est une sous-section (ex: '3.1 The Work')."""
        pattern = r"^(\d+(?:\.\d+)+)\s*(.*?)$"
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1)
        return None

    def _is_section_start(self, line: str) -> Optional[str]:
        """Détecte si une ligne commence une nouvelle section."""
        original_line = line
        line = line.strip()

        # Exclure les lignes de pagination et de version
        if re.match(
            r"^(?:Page|Version|Document|File|Date|Time|Author|Status|Confidential|Proprietary|Copyright|All rights reserved|©|\(c\)|\(C\)|\[|\]|\||\+|=|_|\s*$)",
            line,
            re.IGNORECASE,
        ):
            return None

        # Vérifier d'abord si la ligne contient un numéro
        if not re.search(r"\d+", line):
            return None

        # SUPER PATTERN pour les cas critiques comme "### **10.2** MISSION MINISTERIELLE PME/PMI"
        # Ce pattern est extrêmement permissif pour éviter les problèmes de caractères spéciaux
        critical_pattern = r"^#+\s*\*\*(\d+(?:\.\d+)*)\*\*\s+(.+)$"
        if match := re.match(critical_pattern, line):
            section_number = match.group(1)
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number
            
        # Pattern spécifique pour le cas "**4.9** DELAI D'ETABLISSEMENT..."
        if re.match(r"^\*\*(\d+(?:\.\d+)*)\*\*\s+[A-Z]", line):
            section_number = re.match(r"^\*\*(\d+(?:\.\d+)*)\*\*", line).group(1)
            title = line.split("**", 2)[2].strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les sous-sections avec tiret (ex: "- 3.1 The Work")
        pattern_dash_subsection = r"^-\s*(\d+(?:\.\d+)+)\s*(.*?)$"
        if match := re.match(pattern_dash_subsection, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec marqueurs de formatage
        # Ex: "# 1. Formation", "- 2. Definitions", "3. Work"
        pattern_chapter = r"^(?:[#*-]+\s*)?(\d+)\.\s+(.*?)$"
        if match := re.match(pattern_chapter, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec formatage Markdown
        # Ex: "## **1. Formation**", "### **2. Definitions**"
        pattern_markdown = r"^#+\s*\*\*(\d+)\.\s+(.*?)\*\*$"
        if match := re.match(pattern_markdown, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Nouveau pattern pour les sous-sections avec formatage Markdown
        # Ex: "# 6.4 Defects", "#### 6.1 Rejection"
        pattern_markdown_subsection = r"^#+\s*(\d+(?:\.\d+)+)\s*(.*?)$"
        if match := re.match(pattern_markdown_subsection, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec formatage spécial
        # Ex: "**Titre du chapitre**: **6. Procedure for Equipment Design...**"
        pattern_special = (
            r"\*\*Titre du chapitre\*\*:\s*\*\*(\d+(?:\.\d+)*\.?)\s+(.*?)\*\*"
        )
        if match := re.match(pattern_special, line):
            section_number = self._normalize_section_number(match.group(1))
            if section_number.endswith("."):
                section_number = section_number[:-1]
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec format "### FORCE MAJEURE 11."
        # Ex: "### FORCE MAJEURE 11.", "#### 18. CONTRACTOR CLAIMS"
        pattern_title_number = r"^#+\s*([A-Z\s]+)\s*(\d+)\.?$"
        if match := re.match(pattern_title_number, line):
            section_number = self._normalize_section_number(match.group(2))
            title = match.group(1).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec format "### 11. FORCE MAJEURE"
        # Ex: "### 11. FORCE MAJEURE", "#### 18. CONTRACTOR CLAIMS"
        pattern_number_title = r"^#+\s*(\d+)\.\s*([A-Z\s]+)$"
        if match := re.match(pattern_number_title, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres de chapitre avec numéro collé à la fin
        # Ex: "## **CONFIDENTIALITY AND INTELLECTUAL PROPERTY RIGHTS12.**"
        pattern_title_number_attached = r"^#+\s*\*\*([A-Z\s]+)(\d+)\.?\*\*$"
        if match := re.match(pattern_title_number_attached, line):
            section_number = self._normalize_section_number(match.group(2))
            title = match.group(1).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres avec numéro de section en gras suivi d'un titre normal
        # Ex: "### **2.8** RETENUE DE GARANTIE" ou """### **2.8** RETENUE DE GARANTIE"""
        pattern_bold_number_title = r'^"{0,3}#+\s*\*\*(\d+(?:\.\d+)*)\*\*\s+([A-Z0-9\s\'\-àáâäèéêëìíîïòóôöùúûüçÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÇ_,;:()]+)"{0,3}$'
        if match := re.match(pattern_bold_number_title, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres où numéro et titre complet sont en gras dans le même bloc
        # Ex: "# **8 RECEPTIONS PARTIELLES - ACHEVEMENT DE LA MISSION**"
        pattern_bold_number_and_title = r'^"{0,3}#+\s*\*\*(\d+(?:\.\d+)*)\s+([A-Z0-9\s\'\-àáâäèéêëìíîïòóôöùúûüçÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÇ_,;:()]+)\*\*"{0,3}$'
        if match := re.match(pattern_bold_number_and_title, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les titres qui commencent directement par le numéro en gras (sans #)
        # Ex: "**4.9** DELAI D'ETABLISSEMENT DES DOCUMENTS DE MAITRISE D'ŒUVRE - PENALITES"
        pattern_direct_bold_number = r'^"{0,3}\*\*(\d+(?:\.\d+)*)\*\*\s+([A-Z0-9\s\'\-àáâäèéêëìíîïòóôöùúûüçÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÇ_,;:()]+)"{0,3}$'
        if match := re.match(pattern_direct_bold_number, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        # Pattern pour les numéros de section avec espace (ex: "8 1")
        pattern_space_number = r"^(\d+)\s+(\d+)\s*(.*?)$"
        if match := re.match(pattern_space_number, line):
            section_number = f"{match.group(1)}.{match.group(2)}"
            return self._normalize_section_number(section_number)

        # Nouveau pattern pour les sections qui commencent simplement par un numéro sans tiret
        # Ex: "4.33 eygdyegde"
        pattern_simple_number = r"^(\d+(?:\.\d+)*)\s*(.*?)$"
        if match := re.match(pattern_simple_number, line):
            section_number = self._normalize_section_number(match.group(1))
            title = match.group(2).strip()
            if title:
                self.section_titles[section_number] = title
            return section_number

        return None

    def _get_hierarchy(self, section_number: str) -> List[str]:
        """Retourne la hiérarchie complète pour une section."""
        if not section_number:
            return []

        parts = section_number.split(".")
        chapter_number = parts[0]

        # Si c'est un chapitre
        if len(parts) == 1:
            if chapter_number in self.section_titles:
                return [
                    f"{chapter_number} (**{chapter_number}. {self.section_titles[chapter_number]}**)"
                ]
            return [chapter_number]

        # Si c'est une sous-section
        if chapter_number in self.section_titles:
            return [
                f"{chapter_number} (**{chapter_number}. {self.section_titles[chapter_number]}**) -> {section_number}"
            ]
        return [chapter_number, "->", section_number]

    def _get_parent_section(self, section_number: str) -> Optional[str]:
        """Retourne le numéro de la section parente."""
        if not section_number or "." not in section_number:
            return None
        return ".".join(section_number.split(".")[:-1])

    def _extract_chapter_title(self, line: str) -> Optional[Tuple[str, str]]:
        """Extrait le numéro et le titre d'un chapitre."""
        # Pattern pour détecter les titres de chapitre
        # Exemple: "**Titre du chapitre**: **6. Procedure for Equipment Design...**"
        pattern = r"\*\*Titre du chapitre\*\*:\s*\*\*(\d+(?:\.\d+)*\.?)\s+(.*?)\*\*"
        match = re.match(pattern, line)
        if match:
            section_number = match.group(1)
            if section_number.endswith("."):
                section_number = section_number[:-1]
            return section_number, match.group(2).strip()
        return None

    def _is_table_start(self, line: str) -> bool:
        """Détecte si une ligne marque le début d'un tableau."""
        # Patterns pour détecter différents formats de tableaux
        table_patterns = [
            r"^\s*\|.*\|\s*$",  # Tableau Markdown
            r"^\s*\+[-+]+\+\s*$",  # Ligne de séparation de tableau
            r"^\s*[-]+\s*$",  # Ligne de séparation simple
            r"^\s*[_]+\s*$",  # Ligne de séparation avec underscore
            r"^\s*[=]+\s*$",  # Ligne de séparation avec égal
            r"^\s*[A-Za-z0-9\s]+\s*\|\s*[A-Za-z0-9\s]+\s*$",  # Ligne avec séparateur vertical
            r"^\s*[A-Za-z0-9\s]+\s*[,;]\s*[A-Za-z0-9\s]+\s*$",  # Ligne avec séparateur horizontal
            r"^\s*\|.*$",  # Ligne commençant par un séparateur vertical
            r"^\s*\+.*$",  # Ligne commençant par un +
            r"^\s*[-_=]+\s*$",  # Ligne de séparation avec différents caractères
        ]

        # Vérifier chaque pattern
        for pattern in table_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _is_table_end(self, line: str) -> bool:
        """Détecte si une ligne marque la fin d'un tableau."""
        # Un tableau se termine quand :
        # 1. On trouve une ligne vide ET la ligne suivante n'est pas un tableau
        # 2. On trouve un nouveau numéro de section
        # 3. On trouve une ligne qui n'est pas un tableau ET qui n'est pas une continuation du tableau
        if not line.strip():
            return False

        if self._is_section_start(line) is not None:
            return True

        if not self._is_table_start(line):
            # Vérifier si c'est une continuation du tableau
            # (par exemple, une cellule qui continue sur plusieurs lignes)
            if not re.match(r"^\s*\|.*$", line) and not re.match(r"^\s*\+.*$", line):
                return True

        return False

    def display_chunks(self, chunks: List[Chunk]) -> None:
        """Affiche les chunks de manière lisible."""
        print("\n📝 Affichage des chunks:")
        print("=" * 80)

        for i, chunk in enumerate(chunks, 1):
            print("\n" + "-" * 40)
            print(f"\nChunk {i}/{len(chunks)}")
            print("-" * 40)

            if chunk.section_number:
                print(f"Section: {chunk.section_number}")
                if chunk.hierarchy:
                    print("Hiérarchie complète:", " ".join(chunk.hierarchy))

            print("\nContenu:")
            print(chunk.content)
            print("-" * 40)

        print(f"\n📊 Nombre total de chunks: {len(chunks)}")

    def _is_empty_section_chunk(self, chunk: Chunk) -> bool:
        """
        Determines if a chunk is just a section title with no actual content.
        
        Args:
            chunk: The Chunk object to check
            
        Returns:
            bool: True if the chunk only contains a title and no content
        """
        # Skip if chunk has no section number
        if not chunk.section_number:
            return False
            
        # Get content lines
        lines = chunk.content.split('\n')
        if not lines:
            return True
            
        # If the chunk has only 1-2 lines, it's likely just a title
        if len(lines) <= 2:
            # Check if the content is just a title
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                # Check if this line contains the section number
                if chunk.section_number in stripped:
                    # This is likely just the title line
                    continue
                else:
                    # Found content beyond the title
                    return False
                    
            # If we got here, all lines were either empty or contain the section number
            return True
            
        # Consider it empty if it has only the title and blank lines
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) <= 1:
            return True
            
        # If the content is very short (just a few characters beyond the section title)
        content_text = ' '.join(line for line in lines if line.strip() and chunk.section_number not in line)
        if len(content_text.strip()) < 10:
            return True
            
        return False

    def split(self, text: str) -> List[Chunk]:
        """Divise le texte en chunks intelligents."""
        print("\n🔍 Découpage intelligents du texte...")
        start_time = time.time()

        # Découpe ligne par ligne et détecte tous les débuts de section
        lines = text.split('\n')
        section_markers = []
        
        # Trouver toutes les lignes qui sont des débuts de section
        for i, line in enumerate(lines):
            section_number = self._is_section_start(line.strip())
            if section_number:
                section_markers.append((i, section_number))
        
        print(f"🔍 Sections détectées: {len(section_markers)}")
        
        # Si aucune section n'est détectée, retourner un seul chunk
        if not section_markers:
            chunk = Chunk(
                content=text,
                section_number=None,
                document_title=self.document_title,
                hierarchy=[],
            )
            return [chunk]
            
        # Créer des chunks à partir des marqueurs de section
        chunks = []
        empty_section_titles = {}
        
        for i in range(len(section_markers)):
            start_idx, section_number = section_markers[i]
            
            # Définir la fin de la section actuelle
            if i < len(section_markers) - 1:
                end_idx = section_markers[i + 1][0]
            else:
                end_idx = len(lines)
                
            # Créer le contenu du chunk
            content = '\n'.join(lines[start_idx:end_idx])
            
            # Récupérer le titre s'il existe
            section_title = self.section_titles.get(section_number, "")
            
            # Créer le chunk avec les métadonnées
            chunk = Chunk(
                content=content,
                section_number=section_number,
                document_title=self.document_title,
                hierarchy=self._get_hierarchy(section_number),
                parent_section=self._get_parent_section(section_number),
                chapter_title=self.section_titles.get(section_number.split('.')[0] if section_number and '.' in section_number else section_number, None)
            )
            
            # Vérifier si c'est un chunk vide (juste un titre de section)
            if self._is_empty_section_chunk(chunk):
                # Stocker le titre pour référence ultérieure
                empty_section_titles[section_number] = section_title
                print(f"🔍 Section vide détectée: {section_number} - {section_title}")
            else:
                chunks.append(chunk)

        print(f"\n✅ Découpage terminé en {time.time() - start_time:.2f} secondes")
        print(f"📦 Nombre de chunks créés: {len(chunks)}")
        if empty_section_titles:
            print(f"📑 Sections vides ignorées: {len(empty_section_titles)}")

        # Afficher les chunks
        self.display_chunks(chunks)

        return chunks
