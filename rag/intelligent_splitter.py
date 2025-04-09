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

    def _is_section_start(self, line: str) -> Optional[str]:
        """Détecte si une ligne commence par un numéro de section."""
        line = line.strip()
        # Pattern pour détecter les numéros de section dans différents formats
        # Exemples : 
        # - "1.11"
        # - "- 1.11"
        # - "* 1.11"
        # - "1.11 **"Incoming Inspection Committee"**"
        # - "5."
        # - "## **5.**"
        # - "## **7. Contract Price**"
        # - "## CONFIDENTIALITY AND INTELLECTUAL PROPERTY RIGHTS 12."
        # - "### 1. APPLICABILITY AND DEFINITIONS"
        
        # Pattern pour les numéros au début
        pattern_start = r'^(?:[*-]|\#+)?\s*\**\s*(\d+(?:\.\d+)*\.?)\**'
        # Pattern pour les numéros à la fin
        pattern_end = r'^(?:[*-]|\#+)?\s*\**.*?\s+(\d+(?:\.\d+)*\.?)\s*$'
        
        # Essayer d'abord le pattern de début
        match = re.match(pattern_start, line)
        if match:
            section_number = match.group(1)
            if section_number.endswith('.'):
                section_number = section_number[:-1]
            return section_number
            
        # Si pas de match, essayer le pattern de fin
        match = re.match(pattern_end, line)
        if match:
            section_number = match.group(1)
            if section_number.endswith('.'):
                section_number = section_number[:-1]
            return section_number
            
        return None

    def _update_hierarchy(self, section_number: str, title: str):
        """Met à jour la hiérarchie des sections avec le nouveau titre."""
        self.section_titles[section_number] = title
        
        # Construire la hiérarchie complète avec les titres uniquement pour les parties principales
        parts = section_number.split('.')
        hierarchy = []
        current_path = []
        
        for i, part in enumerate(parts):
            current_path.append(part)
            current_section = '.'.join(current_path)
            # On n'inclut le titre que pour les parties principales (niveau 1)
            if i == 0 and current_section in self.section_titles:
                hierarchy.append(f"{current_section} ({self.section_titles[current_section]})")
            else:
                hierarchy.append(current_section)
        
        return hierarchy

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

    def display_chunks(self, chunks: List[Chunk]) -> None:
        """Affiche les chunks de manière lisible."""
        print("\n📝 Affichage des chunks:")
        print("=" * 80)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}/{len(chunks)}")
            print("-" * 40)
            
            if chunk.section_number:
                print(f"Section: {chunk.section_number}")
                if chunk.hierarchy:
                    print("Hiérarchie complète:", " -> ".join(chunk.hierarchy))
                if chunk.parent_section:
                    print(f"Section parente: {chunk.parent_section}")
            
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
        
        lines = text.split('\n')
        print(f"📊 Document analysé: {len(lines)} lignes")
        
        for line in tqdm(lines, desc="Traitement des lignes", unit="ligne"):
            # Vérifier si c'est un titre de chapitre
            chapter_info = self._extract_chapter_title(line)
            if chapter_info:
                section_number, title = chapter_info
                self.section_titles[section_number] = title
                continue
                
            section_number = self._is_section_start(line)
            
            # Si on trouve un nouveau numéro de section, créer le chunk précédent
            if section_number and current_lines:
                # Extraire le titre de la section actuelle
                if current_section:
                    current_title = ' '.join(current_lines[0].split()[1:]) if current_lines else None
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
        
        # Ajouter le dernier chunk
        if current_lines:
            if current_section:
                current_title = ' '.join(current_lines[0].split()[1:]) if current_lines else None
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