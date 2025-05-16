import os
import platform
from pathlib import Path
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from utils.logger import setup_logger

# Load environment variables
load_dotenv("config.env")

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class TextVectorizer:
    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
    ):
        # Charger les variables d'environnement
        load_dotenv("config.env")

        # Utiliser les valeurs de config.env ou les valeurs par défaut
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.cache_dir = Path(
            cache_dir or os.getenv("CACHE_DIR", "offline_models/embeddings_cache")
        )
        self.models_dir = Path(os.getenv("MODELS_DIR", "offline_models/hf"))
        self.embeddings_dir = Path(
            os.getenv("EMBEDDINGS_DIR", "offline_models/embeddings")
        )
        self.use_offline_models = (
            os.getenv("USE_OFFLINE_MODELS", "true").lower() == "true"
        )

        # Créer les dossiers nécessaires
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialisation du TextVectorizer avec le modèle '{self.model_name}'"
        )
        logger.debug(f"Répertoire de cache: {self.cache_dir}")
        logger.debug(f"Répertoire de modèles: {self.models_dir}")
        logger.debug(f"Mode hors ligne: {self.use_offline_models}")

        # Détection du device
        is_apple_silicon = (
            platform.processor() == "arm" and platform.system() == "Darwin"
        )
        use_mps = os.getenv("USE_MPS", "true").lower() == "true"

        if is_apple_silicon and torch.backends.mps.is_available() and use_mps:
            self.device = torch.device("mps")
            logger.info("🎮 Using MPS (Metal Performance Shaders) for embeddings")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"💻 Using {self.device} for embeddings")

        # Charger le modèle avec gestion des erreurs et fallback
        self._load_model()

    def _load_model(self):
        """Charge le modèle avec gestion des erreurs et fallback"""
        # Sauvegarde des variables d'environnement originales
        original_env = {}
        env_vars = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
        for var in env_vars:
            original_env[var] = os.environ.get(var)

        try:
            # Essayer d'abord en mode hors ligne si demandé
            if self.use_offline_models:
                logger.info(
                    "🔒 Mode hors ligne activé - Tentative d'utilisation des modèles locaux"
                )
                # Configurer l'environnement pour le mode hors ligne
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                os.environ["NO_PROXY"] = "*"
                os.environ["http_proxy"] = ""
                os.environ["https_proxy"] = ""

                try:
                    logger.info(
                        f"🔄 Chargement du modèle {self.model_name} depuis {self.models_dir} (mode hors ligne)"
                    )
                    self.model = SentenceTransformer(
                        self.model_name,
                        cache_folder=str(self.models_dir),
                        local_files_only=True,
                    )
                    logger.info(f"✅ Modèle chargé avec succès en mode hors ligne")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Impossible de charger le modèle en mode hors ligne: {str(e)}"
                    )
                    logger.info(
                        "🌐 Passage en mode en ligne pour télécharger le modèle"
                    )

                    # Réinitialiser les variables d'environnement pour passer en mode en ligne
                    for var in env_vars:
                        if original_env[var] is None:
                            if var in os.environ:
                                del os.environ[var]
                        else:
                            os.environ[var] = original_env[var]

                    # Télécharger le modèle en mode en ligne
                    logger.info(
                        f"📥 Téléchargement du modèle {self.model_name} vers {self.models_dir}"
                    )
                    self.model = SentenceTransformer(
                        self.model_name, cache_folder=str(self.models_dir)
                    )
                    logger.info(f"✅ Modèle téléchargé avec succès")

                    # Recommander à l'utilisateur d'exécuter la fonction de téléchargement
                    logger.info(
                        "💡 Pour utiliser ce modèle en mode hors ligne à l'avenir, exécutez download_models_for_offline_use()"
                    )
            else:
                # Chargement normal en mode en ligne
                logger.info(
                    f"🌐 Chargement du modèle {self.model_name} en mode en ligne"
                )
                self.model = SentenceTransformer(
                    self.model_name, cache_folder=str(self.models_dir)
                )
                logger.info(f"✅ Modèle chargé avec succès")

            # Déplacer le modèle sur le bon device
            self.model.to(self.device)
            logger.info(f"✅ Modèle déplacé vers {self.device}")

            # Optimisations pour MPS/CUDA
            use_half_precision = (
                os.getenv("USE_HALF_PRECISION", "true").lower() == "true"
            )
            if use_half_precision and (
                self.device.type == "mps" or self.device.type == "cuda"
            ):
                self.model.half()  # Utiliser la précision FP16 pour de meilleures performances
                logger.debug(f"⚡ Optimisation FP16 appliquée pour {self.device.type}")

        except Exception as e:
            logger.error(
                f"❌ Erreur fatale lors du chargement du modèle {self.model_name}: {str(e)}"
            )
            raise
        finally:
            # Restaurer les variables d'environnement originales si le mode hors ligne est activé
            if self.use_offline_models:
                for var in env_vars:
                    if original_env[var] is None:
                        if var in os.environ:
                            del os.environ[var]
                    else:
                        os.environ[var] = original_env[var]

    @classmethod
    def download_models_for_offline_use(cls, model_name=None, all_models=True):
        """Télécharge les modèles pour une utilisation hors ligne

        Télécharge et met en cache les modèles d'embeddings pour une utilisation future
        sans connexion internet. Supporte deux modèles principaux :
        
        1. sentence-transformers/all-mpnet-base-v2 :
           - Contexte: 768 tokens
           - Précision: Excellente pour les textes courts
           - Utilisé pour le chunking sémantique
        
        2. BAAI/bge-m3 :  
           - Contexte: Peut gérer des textes plus longs (1024+)
           - Précision: Supérieure à all-mpnet-base-v2 sur les textes juridiques longs
           - Utilisé pour l'embedding de documents

        Args:
            model_name: Nom du modèle à télécharger (optionnel)
            all_models: Si True, télécharge aussi les autres modèles utilisés dans l'application
        """
        # Charger les variables d'environnement
        load_dotenv("config.env")

        # Liste des modèles à télécharger
        models_to_download = []

        # Le modèle principal (embeddings)
        main_model = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        models_to_download.append(main_model)

        # Ajouter les autres modèles utilisés dans l'application
        if all_models:
            # Modèle utilisé par TextChunker
            models_to_download.append("sentence-transformers/all-mpnet-base-v2")
            # Ajouter d'autres modèles ici si nécessaire

        # Répertoire des modèles
        models_dir = Path(os.getenv("MODELS_DIR", "offline_models/hf"))
        models_dir.mkdir(parents=True, exist_ok=True)

        # Désactiver temporairement le mode hors ligne
        original_env = {}
        env_vars = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
        for var in env_vars:
            original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        # Résultats du téléchargement
        results = {}

        try:
            # Télécharger chaque modèle
            for model in models_to_download:
                logger.info(
                    f"📥 Téléchargement du modèle {model} pour utilisation hors ligne"
                )
                logger.info(f"Destination: {models_dir}")

                try:
                    # Télécharger le modèle
                    model_instance = SentenceTransformer(
                        model, cache_folder=str(models_dir)
                    )
                    # Forcer le téléchargement du tokenizer et des fichiers de configuration
                    _ = model_instance.tokenizer

                    # Accès à d'autres composants selon le type de modèle
                    try:
                        _ = model_instance.auto_model.config
                    except AttributeError:
                        # Certains modèles n'ont pas d'attribut auto_model
                        logger.debug(
                            f"Le modèle {model} n'a pas d'attribut auto_model, ignoré"
                        )

                    logger.info(f"✅ Modèle {model} téléchargé avec succès")
                    results[model] = True
                except Exception as e:
                    logger.error(
                        f"❌ Erreur lors du téléchargement du modèle {model}: {str(e)}"
                    )
                    results[model] = False

            # Message de succès global
            success_count = sum(1 for v in results.values() if v)
            if success_count == len(models_to_download):
                logger.info(
                    f"✅ Tous les modèles ({len(models_to_download)}) ont été téléchargés avec succès"
                )
            else:
                logger.warning(
                    f"⚠️ {success_count}/{len(models_to_download)} modèles téléchargés avec succès"
                )

            logger.info(
                f"💡 Pour utiliser ces modèles, assurez-vous que USE_OFFLINE_MODELS=true dans config.env"
            )
            return results
        except Exception as e:
            logger.error(f"❌ Erreur lors du téléchargement des modèles: {str(e)}")
            return results
        finally:
            # Restaurer les variables d'environnement originales
            for var in env_vars:
                if original_env[var] is not None:
                    os.environ[var] = original_env[var]

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
