from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from .intelligent_splitter import Chunk

@dataclass
class HierarchicalGroup:
    section_number: str
    chunks: List[Chunk]
    subgroups: Dict[str, 'HierarchicalGroup']
    level: int

class HierarchicalGrouper:
    def __init__(self):
        self.root_groups: Dict[str, HierarchicalGroup] = {}
        self.depth_stats = defaultdict(int)

    def _get_section_level(self, section_number: str) -> int:
        """Détermine le niveau hiérarchique d'une section."""
        if not section_number:
            return 0
        return len(section_number.split('.'))

    def _get_parent_section(self, section_number: str) -> str:
        """Retourne le numéro de section parent."""
        if not section_number or '.' not in section_number:
            return None
        return '.'.join(section_number.split('.')[:-1])

    def group_chunks(self, chunks: List[Chunk]) -> Dict[str, HierarchicalGroup]:
        """Groupe les chunks selon leur structure hiérarchique."""
        # Réinitialiser les groupes
        self.root_groups = {}
        
        # Premier passage : créer tous les groupes
        for chunk in chunks:
            if not chunk.section_number:
                continue
                
            section_number = chunk.section_number
            level = self._get_section_level(section_number)
            
            # Mettre à jour les statistiques de profondeur
            self.depth_stats[level] += 1
            
            # Créer le groupe s'il n'existe pas
            if section_number not in self.root_groups:
                self.root_groups[section_number] = HierarchicalGroup(
                    section_number=section_number,
                    chunks=[],
                    subgroups={},
                    level=level
                )
            
            # Ajouter le chunk au groupe
            self.root_groups[section_number].chunks.append(chunk)
        
        # Deuxième passage : organiser la hiérarchie
        final_groups = {}
        for section_number, group in self.root_groups.items():
            parent_section = self._get_parent_section(section_number)
            
            if parent_section:
                # Si le parent existe, ajouter comme sous-groupe
                if parent_section in self.root_groups:
                    self.root_groups[parent_section].subgroups[section_number] = group
            else:
                # Si pas de parent, c'est un groupe racine
                final_groups[section_number] = group
        
        return final_groups

    def get_depth_statistics(self) -> Dict[int, int]:
        """Retourne des statistiques sur la profondeur des sections."""
        return dict(self.depth_stats) 