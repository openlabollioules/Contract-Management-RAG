from typing import List, Dict, Optional
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

class IntelligentSplitter:
    def __init__(self, document_title: Optional[str] = None):
        self.document_title = document_title
        self.chunk_count = 0

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
        pattern = r'^(?:[*-]|\#+)?\s*\**\s*(\d+(?:\.\d+)*\.?)\**'
        match = re.match(pattern, line)
        if match:
            # Si le numéro se termine par un point, on le retire pour la cohérence
            section_number = match.group(1)
            if section_number.endswith('.'):
                section_number = section_number[:-1]
            return section_number
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
        
        lines = text.split('\n')
        print(f"📊 Document analysé: {len(lines)} lignes")
        
        for line in tqdm(lines, desc="Traitement des lignes", unit="ligne"):
            section_number = self._is_section_start(line)
            
            # Si on trouve un nouveau numéro de section, créer le chunk précédent
            if section_number and current_lines:
                # Créer le chunk avec le contenu accumulé
                chunk = Chunk(
                    content='\n'.join(current_lines),
                    section_number=current_section
                )
                chunks.append(chunk)
                self.chunk_count += 1
                
                # Réinitialiser pour le nouveau chunk
                current_lines = []
                current_section = section_number
            
            # Si c'est le premier chunk et qu'il n'y a pas de numéro de section
            elif not current_section and not section_number:
                current_section = None
            
            current_lines.append(line)
        
        # Ajouter le dernier chunk
        if current_lines:
            chunk = Chunk(
                content='\n'.join(current_lines),
                section_number=current_section
            )
            chunks.append(chunk)
            self.chunk_count += 1
        
        print(f"\n✅ Découpage terminé en {time.time() - start_time:.2f} secondes")
        
        # Afficher les chunks
        self.display_chunks(chunks)
        
        return chunks 