#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de téléchargement des modèles marker pour l'extraction PDF.
Ce script télécharge les modèles depuis Hugging Face et les stocke dans le répertoire configuré.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/download_marker_models.log"),
    ],
)
logger = logging.getLogger("download_marker")


def ensure_directory_exists(directory):
    """S'assure qu'un répertoire existe, le crée sinon"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Répertoire vérifié: {directory}")


def check_model_directory(directory):
    """Vérifie si un répertoire de modèle contient des fichiers valides"""
    path = Path(directory)
    if not path.exists():
        return False, "Le répertoire n'existe pas"

    # Vérifier s'il y a des fichiers dans le répertoire
    files = list(path.glob("*"))
    if not files:
        return False, "Le répertoire est vide"

    # Vérifier s'il y a des fichiers modèles spécifiques
    model_files = (
        list(path.glob("*.safetensors"))
        + list(path.glob("*.bin"))
        + list(path.glob("*.pt"))
    )
    config_files = list(path.glob("config.json")) + list(
        path.glob("preprocessor_config.json")
    )

    if not model_files and not config_files:
        return (
            False,
            "Aucun fichier modèle trouvé (.safetensors, .bin, .pt, config.json)",
        )

    # Le modèle semble valide
    return True, f"{len(model_files) + len(config_files)} fichiers modèle trouvés"


def main():
    """Fonction principale pour télécharger les modèles marker"""
    # Charger les variables d'environnement
    load_dotenv("config.env")

    # Créer d'abord le répertoire offline_models de base s'il n'existe pas
    offline_dir = Path("offline_models")
    ensure_directory_exists(offline_dir)

    # Répertoire de base pour les modèles marker
    base_dir = Path(os.getenv("MARKER_DIR", "offline_models/marker"))
    ensure_directory_exists(base_dir)

    # S'assurer que le répertoire des logs existe
    ensure_directory_exists("logs")

    try:
        # Installer huggingface_hub si nécessaire
        try:
            import huggingface_hub

            logger.info(
                f"Module huggingface_hub importé avec succès (version: {huggingface_hub.__version__})"
            )
        except ImportError:
            logger.warning("❌ Module huggingface_hub non trouvé. Installation...")
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"]
            )
            import huggingface_hub

            logger.info(
                f"✅ Module huggingface_hub installé avec succès (version: {huggingface_hub.__version__})"
            )

        from huggingface_hub import snapshot_download

        # Créer les répertoires pour chaque modèle
        model_dirs = {
            "layout": base_dir / "layout",
            "texify": base_dir / "texify",
        }

        # Dépôts corrects des modèles sur Hugging Face pour Marker
        model_repos = {
            "layout": "VikParuchuri/marker_layout_segmenter",
            "texify": "VikParuchuri/marker_texify",
        }

        # Créer les répertoires s'ils n'existent pas
        for dir_path in model_dirs.values():
            ensure_directory_exists(dir_path)

        # Vérifier d'abord si les modèles sont déjà téléchargés
        already_downloaded = {}
        for model_name, dir_path in model_dirs.items():
            is_valid, message = check_model_directory(dir_path)
            already_downloaded[model_name] = is_valid
            if is_valid:
                logger.info(f"✅ Le modèle {model_name} est déjà téléchargé: {message}")
            else:
                logger.info(
                    f"🔄 Le modèle {model_name} doit être téléchargé: {message}"
                )

        # Note sur les modèles Surya
        logger.info(
            "ℹ️ Les modèles de reconnaissance (text_recognition, table_recognition, text_detection, inline_math_detection) "
        )
        logger.info(
            "ℹ️ ne sont pas téléchargés automatiquement car ils ne sont pas disponibles séparément sur Hugging Face."
        )
        logger.info(
            "ℹ️ Ces modèles sont inclus dans le package Surya et seront téléchargés automatiquement lors de l'utilisation."
        )

        # Compter les modèles réussis
        success_count = 0

        # Télécharger chaque modèle
        logger.info(
            "🔄 Début du téléchargement des modèles marker depuis Hugging Face..."
        )
        for model_name, repo_id in model_repos.items():
            # Sauter le téléchargement si le modèle est déjà présent
            if already_downloaded.get(model_name, False):
                logger.info(f"⏭️ Modèle {model_name} déjà téléchargé, ignoré")
                success_count += 1
                continue

            logger.info(f"🔄 Téléchargement du modèle {model_name} depuis {repo_id}...")
            try:
                # Télécharger le modèle
                local_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(model_dirs[model_name]),
                    local_dir_use_symlinks=False,
                )
                logger.info(
                    f"✅ Modèle {model_name} téléchargé avec succès dans {local_path}"
                )
                success_count += 1
            except Exception as e:
                logger.error(
                    f"❌ Erreur lors du téléchargement du modèle {model_name}: {str(e)}"
                )

                # Nettoyer le répertoire en cas d'échec pour permettre une nouvelle tentative
                try:
                    shutil.rmtree(model_dirs[model_name])
                    ensure_directory_exists(model_dirs[model_name])
                    logger.info(
                        f"🧹 Répertoire du modèle {model_name} nettoyé pour future tentative"
                    )
                except Exception as clean_error:
                    logger.error(
                        f"❌ Erreur lors du nettoyage du répertoire {model_name}: {str(clean_error)}"
                    )

        # Vérification finale des modèles téléchargés
        missing_models = []
        for model_name, dir_path in model_dirs.items():
            is_valid, message = check_model_directory(dir_path)
            if not is_valid:
                missing_models.append(model_name)
                logger.warning(
                    f"⚠️ Le modèle {model_name} n'a pas été correctement téléchargé: {message}"
                )
            else:
                logger.info(
                    f"✅ Le modèle {model_name} a été correctement téléchargé: {message}"
                )

        # Résumé
        if success_count == len(model_repos):
            logger.info(
                f"✅ Tous les modèles marker ({len(model_repos)}) ont été téléchargés avec succès"
            )
            return True
        elif success_count > 0:
            logger.warning(
                f"⚠️ {success_count}/{len(model_repos)} modèles marker téléchargés avec succès"
            )
            logger.warning(f"⚠️ Modèles manquants: {missing_models}")
            return True  # Succès partiel si au moins un modèle a été téléchargé
        else:
            logger.error(f"❌ Aucun modèle n'a pu être téléchargé")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement des modèles marker: {str(e)}")
        return False


if __name__ == "__main__":
    # S'assurer que le répertoire des logs existe
    Path("./logs").mkdir(exist_ok=True)

    success = main()
    sys.exit(0 if success else 1)
