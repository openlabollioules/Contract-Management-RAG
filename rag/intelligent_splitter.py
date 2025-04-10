from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import time
import platform
import torch
from tqdm import tqdm

# Détection de l'architecture
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
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
    hierarchy: Optional[Dict[str, str]] = None  # Dictionnaire contenant la hiérarchie complète
    parent_section: Optional[str] = None  # Numéro de la section parente
    chapter_title: Optional[str] = None  # Titre du chapitre

class IntelligentSplitter:
    def __init__(self, document_title: Optional[str] = None):
        self.document_title = document_title
        self.chunk_count = 0
        self.section_hierarchy = {}  # Pour stocker la hiérarchie des sections
        self.section_titles = {}  # Pour stocker les titres des sections

    def _normalize_section_number(self, section_number: str) -> str:
        """Normalise un numéro de section en ajoutant un point entre les chiffres séparés par un espace."""
        # Si le numéro contient un espace, on le remplace par un point
        if ' ' in section_number:
            return section_number.replace(' ', '.')
        return section_number

    def _is_section_start(self, line: str) -> Optional[str]:
        """Détecte si une ligne commence par un numéro de section."""
        line = line.strip()
        
        # Pattern pour les titres de chapitre (X ou X.)
        # Exemples :
        # - "### WARRANTY 10."
        # - "## **5.**"
        # - "## **7. Contract Price**"
        # - "## CONFIDENTIALITY AND INTELLECTUAL PROPERTY RIGHTS 12."
        # - "### 1. APPLICABILITY AND DEFINITIONS"
        # - "5."
        # - "### COMPLIANCE 13."
        # - "# 21. DISPUTE RESOLUTION"
        # - "## CONFIDENTIALITY AND INTELLECTUAL PROPERTY RIGHTS" suivi de "12."
        
        # Pattern pour les titres de chapitre (X ou X.)
        pattern_chapter = r'^\#+\s*\**\s*(\d+)\.?\s*(.*?)\**\s*$|^\#+\s*\**\s*(.*?)\s*(\d+)\.?\s*\**\s*$|^\#+\s*\**\s*(.*?)\s*\**\s*$'
        match = re.match(pattern_chapter, line)
        if match:
            # Si le numéro est au début (premier groupe)
            if match.group(1):
                section_number = match.group(1)
                title = match.group(2).strip()
            # Si le numéro est à la fin (quatrième groupe)
            elif match.group(4):
                section_number = match.group(4)
                title = match.group(3).strip()
            # Si c'est juste un titre (cinquième groupe), on vérifie la ligne suivante
            elif match.group(5):
                return None
            
            # Si c'est un chapitre (un seul chiffre), stocker le titre
            if '.' not in section_number and title:
                self.section_titles[section_number] = title
                
            return section_number
            
        # Pattern pour les numéros seuls (pour la ligne après un titre)
        pattern_number = r'^(\d+)\.\s*$'
        match = re.match(pattern_number, line)
        if match:
            return match.group(1)
            
        # Pattern pour les sections (X.Y, X.Y.Z, etc.)
        pattern_section = r'^\#+\s*\**\s*(\d+(?:\.\d+)+)\s*(.*?)\**\s*$'
        match = re.match(pattern_section, line)
        if match:
            return match.group(1)
            
        # Pattern pour les numéros au début
        pattern_start = r'^(?:[*-]|\#+)?\s*\**\s*(\d+(?:\.\d+)+).*$'
        match = re.match(pattern_start, line)
        if match:
            return match.group(1)
            
        return None

    def _update_hierarchy(self, section_number: str, title: str):
        """Met à jour la hiérarchie des sections avec le nouveau titre."""
        # Construire la hiérarchie complète avec le titre du chapitre principal
        parts = section_number.split('.')
        
        # Si c'est un chapitre (un seul chiffre), retourner le titre complet
        if len(parts) == 1:
            if section_number in self.section_titles:
                return [f"{section_number} (**{section_number}. {self.section_titles[section_number]}**)"]
            return [section_number]
        
        # Pour les sections (X.Y, X.Y.Z, etc.), retourner le titre du chapitre principal -> numéro de section
        chapter_number = parts[0]
        if chapter_number in self.section_titles:
            return [f"{chapter_number} (**{chapter_number}. {self.section_titles[chapter_number]}**) -> {section_number}"]
        return [chapter_number, "->", section_number]

    def _get_parent_section(self, section_number: str) -> Optional[str]:
        """Retourne le numéro de la section parente."""
        if not section_number or '.' not in section_number:
            return None
        return '.'.join(section_number.split('.')[:-1])

    def _extract_chapter_title(self, line: str) -> Optional[Tuple[str, str]]:
        """Extrait le numéro et le titre d'un chapitre."""
        # Pattern pour détecter les titres de chapitre
        # Exemple: "**Titre du chapitre**: **6. Procedure for Equipment Design...**"
        pattern = r'\*\*Titre du chapitre\*\*:\s*\*\*(\d+(?:\.\d+)*\.?)\s+(.*?)\*\*'
        match = re.match(pattern, line)
        if match:
            section_number = match.group(1)
            if section_number.endswith('.'):
                section_number = section_number[:-1]
            return section_number, match.group(2).strip()
        return None

    def _is_table_start(self, line: str) -> bool:
        """Détecte si une ligne marque le début d'un tableau."""
        # Patterns pour détecter différents formats de tableaux
        table_patterns = [
            r'^\s*\|.*\|\s*$',                    # Tableau Markdown
            r'^\s*\+[-+]+\+\s*$',                 # Ligne de séparation de tableau
            r'^\s*[-]+\s*$',                      # Ligne de séparation simple
            r'^\s*[_]+\s*$',                      # Ligne de séparation avec underscore
            r'^\s*[=]+\s*$',                      # Ligne de séparation avec égal
            r'^\s*[A-Za-z0-9\s]+\s*\|\s*[A-Za-z0-9\s]+\s*$',  # Ligne avec séparateur vertical
            r'^\s*[A-Za-z0-9\s]+\s*[,;]\s*[A-Za-z0-9\s]+\s*$',  # Ligne avec séparateur horizontal
            r'^\s*\|.*$',                         # Ligne commençant par un séparateur vertical
            r'^\s*\+.*$',                         # Ligne commençant par un +
            r'^\s*[-_=]+\s*$',                    # Ligne de séparation avec différents caractères
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
            if not re.match(r'^\s*\|.*$', line) and not re.match(r'^\s*\+.*$', line):
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

    def split(self, text: str) -> List[Chunk]:
        """Divise le texte en chunks selon les numéros de section."""
        print("\n🔍 Analyse de la structure du document...")
        start_time = time.time()
        
        chunks = []
        current_lines = []
        current_section = None
        current_title = None
        previous_line = None
        
        lines = text.split('\n')
        print(f"📊 Document analysé: {len(lines)} lignes")
        
        for line in tqdm(lines, desc="Traitement des lignes", unit="ligne"):
            # Vérifier si c'est un titre de chapitre
            chapter_info = self._extract_chapter_title(line)
            if chapter_info:
                section_number, title = chapter_info
                self.section_titles[section_number] = title
                continue
                
            # Si la ligne précédente était un titre et que celle-ci est un numéro
            if previous_line and line.strip():
                pattern_chapter = r'^\#+\s*\**\s*(.*?)\s*\**\s*$'
                match_previous = re.match(pattern_chapter, previous_line)
                pattern_number = r'^(\d+)\.\s*$'
                match_current = re.match(pattern_number, line)
                
                if match_previous and match_current:
                    section_number = match_current.group(1)
                    title = match_previous.group(1).strip()
                    self.section_titles[section_number] = title
                    current_section = section_number
                    current_lines.extend([previous_line, line])
                    previous_line = line
                    continue
            
            section_number = self._is_section_start(line)
            
            # Si on trouve un nouveau numéro de section, créer le chunk précédent
            if section_number and current_lines:
                # Extraire le titre de la section actuelle
                if current_section:
                    hierarchy = self._update_hierarchy(current_section, current_title)
                    
                    # Créer le chunk avec le contenu accumulé
                    chunk = Chunk(
                        content='\n'.join(current_lines),
                        section_number=current_section,
                        document_title=self.document_title,
                        hierarchy=hierarchy,
                        parent_section=self._get_parent_section(current_section),
                        chapter_title=self.section_titles.get(current_section.split('.')[0])
                    )
                    chunks.append(chunk)
                    self.chunk_count += 1
                
                # Réinitialiser pour le nouveau chunk
                current_lines = []
                current_section = section_number
            
            current_lines.append(line)
            previous_line = line
        
        # Ajouter le dernier chunk
        if current_lines:
            if current_section:
                hierarchy = self._update_hierarchy(current_section, current_title)
                
                chunk = Chunk(
                    content='\n'.join(current_lines),
                    section_number=current_section,
                    document_title=self.document_title,
                    hierarchy=hierarchy,
                    parent_section=self._get_parent_section(current_section),
                    chapter_title=self.section_titles.get(current_section.split('.')[0])
                )
            else:
                chunk = Chunk(
                    content='\n'.join(current_lines),
                    document_title=self.document_title
                )
            chunks.append(chunk)
            self.chunk_count += 1
        
        print(f"\n✅ Découpage terminé en {time.time() - start_time:.2f} secondes")
        
        # Afficher les chunks
        self.display_chunks(chunks)
        
        return chunks 