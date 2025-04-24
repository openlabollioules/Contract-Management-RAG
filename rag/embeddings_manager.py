import platform
from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = "offline_models/embeddings_cache",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialisation du EmbeddingsManager avec le modèle '{model_name}'"
        )
        logger.debug(f"Répertoire de cache: {cache_dir}")

        # Détection du device
        is_apple_silicon = (
            platform.processor() == "arm" and platform.system() == "Darwin"
        )
        if is_apple_silicon and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("🎮 Using MPS (Metal Performance Shaders) for embeddings")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"💻 Using {self.device} for embeddings")

        # Charger le modèle
        logger.debug(f"Chargement du modèle depuis cache_folder=offline_models/hf/hub")
        self.model = SentenceTransformer(
            model_name, cache_folder="offline_models/hf/hub"
        )
        self.model.to(self.device)
        logger.info(f"Modèle chargé et déplacé vers {self.device}")

        # Optimisations pour MPS
        if self.device.type == "mps":
            self.model.half()  # Utiliser la précision FP16 pour de meilleures performances
            logger.debug("Optimisation FP16 appliquée pour MPS")

    def _get_cache_path(self, text: str) -> Path:
        """Génère un chemin de cache unique pour le texte"""
        # Utiliser un hash du texte comme nom de fichier
        text_hash = str(hash(text))
        cache_path = self.cache_dir / f"{text_hash}.npy"
        logger.debug(f"Chemin de cache généré: {cache_path}")
        return cache_path

    def _load_from_cache(self, cache_path: Path) -> np.ndarray:
        """Charge les embeddings depuis le cache"""
        logger.debug(f"Chargement des embeddings depuis le cache: {cache_path}")
        return np.load(str(cache_path))

    def _save_to_cache(self, embeddings: np.ndarray, cache_path: Path):
        """Sauvegarde les embeddings dans le cache"""
        logger.debug(f"Sauvegarde des embeddings dans le cache: {cache_path}")
        np.save(str(cache_path), embeddings)

    def get_embeddings(
        self, texts: List[str], use_cache: bool = True
    ) -> List[np.ndarray]:
        """Génère ou récupère les embeddings pour une liste de textes"""
        logger.info(
            f"Demande d'embeddings pour {len(texts)} textes (use_cache={use_cache})"
        )
        results = []
        texts_to_embed = []
        cache_paths = []

        # Vérifier le cache pour chaque texte
        for text in texts:
            cache_path = self._get_cache_path(text)
            if use_cache and cache_path.exists():
                logger.debug(f"Embeddings trouvés dans le cache")
                results.append(self._load_from_cache(cache_path))
            else:
                logger.debug(
                    f"Embeddings non trouvés dans le cache, ajout à la liste à calculer"
                )
                texts_to_embed.append(text)
                cache_paths.append(cache_path)

        # Générer les nouveaux embeddings si nécessaire
        if texts_to_embed:
            logger.info(f"Génération de {len(texts_to_embed)} nouveaux embeddings")
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                logger.debug(f"Embeddings générés avec succès")

            # Sauvegarder dans le cache et ajouter aux résultats
            for emb, cache_path in zip(embeddings, cache_paths):
                self._save_to_cache(emb, cache_path)
                results.append(emb)

        logger.info(f"Retour de {len(results)} embeddings au total")
        return results

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux embeddings"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        logger.debug(f"Similarité calculée: {similarity:.4f}")
        return similarity
