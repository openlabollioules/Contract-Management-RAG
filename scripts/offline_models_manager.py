#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour gérer les modèles offline de Marker/Surya.
Ce script peut:
1. Capturer les modèles téléchargés automatiquement par Surya et les copier dans notre répertoire offline_models
2. Forcer le téléchargement des modèles manquants
3. Vérifier l'intégrité des modèles locaux
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
        logging.FileHandler("./logs/offline_models_manager.log"),
    ],
)
logger = logging.getLogger("offline_models_manager")

# Chemin du cache Datalab (où Surya stocke ses modèles)
DATALAB_CACHE_DIR = Path.home() / "Library" / "Caches" / "datalab" / "models"

# Liste des modèles Surya avec leurs chemins de cache et noms correspondants dans offline_models
SURYA_MODELS = {
    "layout": {
        "cache_path": DATALAB_CACHE_DIR / "layout",
        "version_folder": "2025_02_18",
        "target_folder": "layout",
    },
    "texify": {
        "cache_path": DATALAB_CACHE_DIR / "texify",
        "version_folder": "2025_02_18",
        "target_folder": "texify",
    },
    "text_recognition": {
        "cache_path": DATALAB_CACHE_DIR / "text_recognition",
        "version_folder": "2025_02_18",
        "target_folder": "text_recognition",
    },
    "table_recognition": {
        "cache_path": DATALAB_CACHE_DIR / "table_recognition",
        "version_folder": "2025_02_18",
        "target_folder": "table_recognition",
    },
    "text_detection": {
        "cache_path": DATALAB_CACHE_DIR / "text_detection",
        "version_folder": "2025_02_28",
        "target_folder": "text_detection",
    },
    "inline_math_detection": {
        "cache_path": DATALAB_CACHE_DIR / "inline_math_detection",
        "version_folder": "2025_02_24",
        "target_folder": "inline_math_detection",
    },
    "ocr_error_detection": {
        "cache_path": DATALAB_CACHE_DIR / "ocr_error_detection",
        "version_folder": "2025_02_18",
        "target_folder": "ocr_error",  # Nom différent dans offline_models
    },
}


def ensure_directory_exists(directory):
    """S'assure qu'un répertoire existe, le crée sinon"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.debug(f"✅ Répertoire vérifié: {directory}")


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
    manifest_files = list(path.glob("manifest.json"))

    if not model_files and not config_files:
        return (
            False,
            "Aucun fichier modèle trouvé (.safetensors, .bin, .pt, config.json)",
        )

    # Le modèle semble valide
    return (
        True,
        f"{len(model_files) + len(config_files) + len(manifest_files)} fichiers modèle trouvés",
    )


def copy_model_files(source_dir, target_dir):
    """Copie les fichiers d'un modèle du cache vers le répertoire offline"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        logger.warning(f"⚠️ Répertoire source {source_dir} n'existe pas")
        return False

    ensure_directory_exists(target_dir)

    files_copied = 0
    try:
        for file in source_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, target_dir)
                logger.debug(f"Copié: {file.name}")
                files_copied += 1

        logger.info(
            f"✅ {files_copied} fichiers copiés de {source_dir} vers {target_dir}"
        )
        return True if files_copied > 0 else False
    except Exception as e:
        logger.error(f"❌ Erreur lors de la copie des fichiers: {str(e)}")
        return False


def force_download_model(model_name):
    """Force le téléchargement d'un modèle spécifique en effectuant un appel à Surya"""
    try:
        # Nous devons importer les modules ici car ils peuvent avoir des dépendances
        # qui ne sont pas installées si on exécute ce script seul
        logger.info(f"🔄 Forçage du téléchargement du modèle {model_name}...")

        if model_name == "text_detection":
            from surya.detection.loader import DetectionModelLoader

            loader = DetectionModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "inline_math_detection":
            from surya.detection.loader import InlineDetectionModelLoader

            loader = InlineDetectionModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "text_recognition":
            from surya.recognition.loader import RecognitionModelLoader

            loader = RecognitionModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "layout":
            from surya.layout.loader import LayoutModelLoader

            loader = LayoutModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "table_recognition":
            from surya.table_rec.loader import TableRecModelLoader

            loader = TableRecModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "texify":
            from surya.texify.loader import TexifyModelLoader

            loader = TexifyModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        elif model_name == "ocr_error_detection":
            from surya.ocr_error.loader import OCRErrorModelLoader

            loader = OCRErrorModelLoader()
            loader.model()
            logger.info(f"✅ Modèle {model_name} téléchargé avec succès")
            return True
        else:
            logger.warning(
                f"⚠️ Modèle {model_name} non supporté pour le téléchargement forcé"
            )
            return False
    except ImportError as e:
        logger.error(f"❌ Module Surya non installé ou incomplet: {str(e)}")
        return False
    except Exception as e:
        logger.error(
            f"❌ Erreur lors du téléchargement forcé du modèle {model_name}: {str(e)}"
        )
        return False


def main():
    """Fonction principale pour gérer les modèles offline"""
    # Charger les variables d'environnement
    load_dotenv("config.env")

    # Créer les répertoires de base s'ils n'existent pas
    offline_dir = Path("offline_models")
    ensure_directory_exists(offline_dir)

    # Répertoire de base pour les modèles marker
    base_dir = Path(os.getenv("MARKER_DIR", "offline_models/marker"))
    ensure_directory_exists(base_dir)

    # S'assurer que le répertoire des logs existe
    ensure_directory_exists("logs")

    # Vérifier si le répertoire de cache Datalab existe
    if not DATALAB_CACHE_DIR.exists():
        logger.warning(
            f"⚠️ Le répertoire de cache Datalab {DATALAB_CACHE_DIR} n'existe pas"
        )
        logger.info(
            "🔄 Vous devez exécuter Surya au moins une fois pour qu'il télécharge les modèles"
        )
        return False

    # Analyser les arguments de ligne de commande
    import argparse

    parser = argparse.ArgumentParser(
        description="Gestionnaire de modèles offline pour Marker/Surya"
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capturer les modèles du cache Datalab vers offline_models",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forcer le téléchargement des modèles manquants",
    )
    parser.add_argument(
        "--check", action="store_true", help="Vérifier l'intégrité des modèles locaux"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Liste des modèles à traiter, séparés par des virgules (défaut: all)",
    )
    args = parser.parse_args()

    # Si aucune action n'est spécifiée, afficher l'aide
    if not (args.capture or args.force_download or args.check):
        parser.print_help()
        return False

    # Déterminer la liste des modèles à traiter
    models_to_process = (
        list(SURYA_MODELS.keys()) if args.models == "all" else args.models.split(",")
    )

    # Vérifier l'intégrité des modèles locaux
    if args.check:
        logger.info("🔍 Vérification de l'intégrité des modèles locaux...")
        valid_models = []
        missing_models = []

        for model_name in models_to_process:
            if model_name not in SURYA_MODELS:
                logger.warning(f"⚠️ Modèle {model_name} inconnu, ignoré")
                continue

            model_info = SURYA_MODELS[model_name]
            target_dir = base_dir / model_info["target_folder"]

            is_valid, message = check_model_directory(target_dir)
            if is_valid:
                logger.info(f"✅ Modèle {model_name} est valide: {message}")
                valid_models.append(model_name)
            else:
                logger.warning(f"⚠️ Modèle {model_name} est invalide: {message}")
                missing_models.append(model_name)

        logger.info(
            f"📊 Résumé: {len(valid_models)}/{len(models_to_process)} modèles valides"
        )
        if missing_models:
            logger.warning(
                f"⚠️ Modèles manquants ou invalides: {', '.join(missing_models)}"
            )

    # Capturer les modèles du cache Datalab vers offline_models
    if args.capture:
        logger.info("🔄 Capture des modèles du cache Datalab vers offline_models...")
        captured_models = []
        failed_models = []

        for model_name in models_to_process:
            if model_name not in SURYA_MODELS:
                logger.warning(f"⚠️ Modèle {model_name} inconnu, ignoré")
                continue

            model_info = SURYA_MODELS[model_name]
            cache_version_dir = model_info["cache_path"] / model_info["version_folder"]
            target_dir = base_dir / model_info["target_folder"]

            if not cache_version_dir.exists():
                logger.warning(
                    f"⚠️ Modèle {model_name} n'existe pas dans le cache: {cache_version_dir}"
                )
                failed_models.append(model_name)
                continue

            success = copy_model_files(cache_version_dir, target_dir)
            if success:
                logger.info(f"✅ Modèle {model_name} capturé avec succès")
                captured_models.append(model_name)
            else:
                logger.error(f"❌ Échec de la capture du modèle {model_name}")
                failed_models.append(model_name)

        logger.info(
            f"📊 Résumé: {len(captured_models)}/{len(models_to_process)} modèles capturés"
        )
        if failed_models:
            logger.warning(f"⚠️ Modèles non capturés: {', '.join(failed_models)}")

    # Forcer le téléchargement des modèles manquants
    if args.force_download:
        logger.info("🔄 Téléchargement forcé des modèles manquants...")

        # On vérifie d'abord quels modèles sont manquants ou invalides
        models_to_download = []
        for model_name in models_to_process:
            if model_name not in SURYA_MODELS:
                logger.warning(f"⚠️ Modèle {model_name} inconnu, ignoré")
                continue

            model_info = SURYA_MODELS[model_name]
            target_dir = base_dir / model_info["target_folder"]

            is_valid, message = check_model_directory(target_dir)
            if not is_valid:
                logger.info(f"🔄 Modèle {model_name} sera téléchargé: {message}")
                models_to_download.append(model_name)
            else:
                logger.info(f"⏭️ Modèle {model_name} existe déjà et est valide")

        # Télécharger les modèles manquants
        downloaded_models = []
        failed_downloads = []

        for model_name in models_to_download:
            success = force_download_model(model_name)
            if success:
                # Après le téléchargement, on capture du cache vers offline_models
                model_info = SURYA_MODELS[model_name]
                cache_version_dir = (
                    model_info["cache_path"] / model_info["version_folder"]
                )
                target_dir = base_dir / model_info["target_folder"]

                if cache_version_dir.exists():
                    copy_success = copy_model_files(cache_version_dir, target_dir)
                    if copy_success:
                        logger.info(
                            f"✅ Modèle {model_name} téléchargé et capturé avec succès"
                        )
                        downloaded_models.append(model_name)
                    else:
                        logger.error(
                            f"❌ Échec de la capture du modèle {model_name} après téléchargement"
                        )
                        failed_downloads.append(model_name)
                else:
                    logger.error(
                        f"❌ Modèle {model_name} téléchargé mais non trouvé dans le cache"
                    )
                    failed_downloads.append(model_name)
            else:
                logger.error(f"❌ Échec du téléchargement du modèle {model_name}")
                failed_downloads.append(model_name)

        logger.info(
            f"📊 Résumé: {len(downloaded_models)}/{len(models_to_download)} modèles téléchargés et capturés"
        )
        if failed_downloads:
            logger.warning(f"⚠️ Modèles non téléchargés: {', '.join(failed_downloads)}")

    return True


if __name__ == "__main__":
    # S'assurer que le répertoire des logs existe
    Path("./logs").mkdir(exist_ok=True)

    success = main()
    sys.exit(0 if success else 1)
