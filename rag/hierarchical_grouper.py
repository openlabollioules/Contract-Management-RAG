from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from utils.logger import setup_logger

from .intelligent_splitter import Chunk

# Configurer le logger pour ce module
logger = setup_logger(__file__)


@dataclass
class HierarchicalGroup:
    section_number: str
    chunks: List[Chunk]
    subgroups: Dict[str, "HierarchicalGroup"]
    level: int


class HierarchicalGrouper:
    def __init__(self, embeddings_manager=None):
        self.root_groups: Dict[str, HierarchicalGroup] = {}
        self.depth_stats = defaultdict(int)
        self.section_titles = {}
        self.embeddings_manager = embeddings_manager
        logger.debug(
            f"HierarchicalGrouper initialisé avec embeddings_manager={embeddings_manager}"
        )

    def _get_section_level(self, section_number: str) -> int:
        """Détermine le niveau hiérarchique d'une section."""
        if not section_number:
            logger.debug(f"Section vide, niveau 0 retourné")
            return 0
        level = len(section_number.split("."))
        logger.debug(f"Niveau de la section {section_number}: {level}")
        return level

    def _get_parent_section(self, section_number: str) -> str:
        """Retourne le numéro de section parent."""
        if not section_number or "." not in section_number:
            logger.debug(f"Pas de parent pour {section_number}")
            return None
        parent = ".".join(section_number.split(".")[:-1])
        logger.debug(f"Parent de {section_number}: {parent}")
        return parent

    def group_chunks(self, chunks: List[Chunk]) -> Dict[str, HierarchicalGroup]:
        """Groupe les chunks selon leur structure hiérarchique."""
        # Réinitialiser les groupes
        self.root_groups = {}
        logger.info(f"Regroupement hiérarchique de {len(chunks)} chunks")

        # Premier passage : créer tous les groupes
        for chunk in chunks:
            if not chunk.section_number:
                logger.debug(
                    f"Chunk sans numéro de section ignoré: {chunk.content[:50]}..."
                )
                continue

            section_number = chunk.section_number
            level = self._get_section_level(section_number)

            # Mettre à jour les statistiques de profondeur
            self.depth_stats[level] += 1

            # Créer le groupe s'il n'existe pas
            if section_number not in self.root_groups:
                logger.debug(f"Création du groupe pour la section {section_number}")
                self.root_groups[section_number] = HierarchicalGroup(
                    section_number=section_number, chunks=[], subgroups={}, level=level
                )

            # Ajouter le chunk au groupe
            self.root_groups[section_number].chunks.append(chunk)
            logger.debug(f"Chunk ajouté au groupe {section_number}")

        # Deuxième passage : organiser la hiérarchie
        final_groups = {}
        logger.info(
            f"Organisation de la hiérarchie pour {len(self.root_groups)} groupes"
        )
        for section_number, group in self.root_groups.items():
            parent_section = self._get_parent_section(section_number)

            if parent_section:
                # Si le parent existe, ajouter comme sous-groupe
                if parent_section in self.root_groups:
                    logger.debug(
                        f"Section {section_number} ajoutée comme sous-groupe de {parent_section}"
                    )
                    self.root_groups[parent_section].subgroups[section_number] = group
            else:
                # Si pas de parent, c'est un groupe racine
                logger.debug(f"Section {section_number} ajoutée comme groupe racine")
                final_groups[section_number] = group

        logger.info(f"Regroupement terminé: {len(final_groups)} groupes racines")
        return final_groups

    def get_depth_statistics(self) -> Dict[int, int]:
        """Retourne des statistiques sur la profondeur des sections."""
        logger.debug(f"Statistiques de profondeur: {dict(self.depth_stats)}")
        return dict(self.depth_stats)

    def _update_hierarchy(self, section_number: str, title: str):
        """Met à jour la hiérarchie des sections avec le nouveau titre."""
        # Construire la hiérarchie complète avec le titre du chapitre principal
        parts = section_number.split(".")
        logger.debug(
            f"Mise à jour de la hiérarchie pour {section_number} avec le titre {title}"
        )

        # Si c'est un chapitre (un seul chiffre), retourner le titre complet
        if len(parts) == 1:
            if section_number in self.section_titles:
                return [
                    f"{section_number} (**{section_number}. {self.section_titles[section_number]}**)"
                ]
            return [section_number]

        # Pour les sections (X.Y, X.Y.Z, etc.), retourner le titre du chapitre principal -> numéro de section
        chapter_number = parts[0]
        if chapter_number in self.section_titles:
            return [
                f"{chapter_number} (**{chapter_number}. {self.section_titles[chapter_number]}**) -> {section_number}"
            ]
        return [chapter_number, "->", section_number]
