#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de téléchargement de tous les modèles pour l'utilisation hors ligne.
Ce script télécharge:
1. Les modèles d'embeddings (BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2)
2. Les modèles marker pour l'extraction de PDF
3. Les ressources NLTK si nécessaire
4. Autres dépendances nécessaires pour le fonctionnement hors ligne
"""

import os
import sys
import subprocess
import importlib.util
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key

# Ajouter le répertoire parent au PYTHONPATH pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configuration du logger pour ce script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/download_all_models.log"),
    ],
)
logger = logging.getLogger("download_all_models")

def ensure_directory_exists(directory):
    """S'assure qu'un répertoire existe, le crée sinon"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Répertoire vérifié: {directory}")

def update_env_file(file_path, key, value):
    """Met à jour une variable dans un fichier .env"""
    try:
        # Charger le fichier
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Chercher et modifier la ligne correspondante
        updated = False
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                if line.split('=')[0].strip() == key:
                    lines[i] = f'{key}="{value}"\n'
                    updated = True
                    break
        
        # Si la clé n'existe pas, l'ajouter
        if not updated:
            lines.append(f'{key}="{value}"\n')
        
        # Écrire les modifications
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        logger.info(f"✅ Fichier {file_path} mis à jour: {key}={value}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de la mise à jour de {file_path}: {str(e)}")
        return False

def download_embeddings_models():
    """Télécharge les modèles d'embeddings"""
    logger.info("🔄 Téléchargement des modèles d'embeddings...")
    
    try:
        # Importer la classe TextVectorizer
        from document_processing.text_vectorizer import TextVectorizer
        
        # Télécharger tous les modèles d'embeddings
        result = TextVectorizer.download_models_for_offline_use(all_models=True)
        
        # Vérifier le résultat
        success_count = sum(1 for v in result.values() if v)
        if success_count == len(result):
            logger.info(f"✅ Tous les modèles d'embeddings ({len(result)}) téléchargés avec succès")
            for model, status in result.items():
                logger.info(f"  - {model}: {'✅' if status else '❌'}")
            return True
        else:
            logger.warning(f"⚠️ {success_count}/{len(result)} modèles d'embeddings téléchargés")
            for model, status in result.items():
                logger.info(f"  - {model}: {'✅' if status else '❌'}")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement des modèles d'embeddings: {str(e)}")
        return False

def download_marker_models():
    """Télécharge les modèles marker pour l'extraction de PDF"""
    logger.info("🔄 Téléchargement des modèles marker...")
    
    try:
        # Vérifier si le script marker existe
        marker_script = Path("scripts/download_marker_models.py")
        if marker_script.exists():
            # Exécuter le script marker
            logger.info("Exécution du script download_marker_models.py")
            result = subprocess.run(
                [sys.executable, str(marker_script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Modèles marker téléchargés avec succès")
                return True
            else:
                logger.error(f"❌ Erreur lors du téléchargement des modèles marker: {result.stderr}")
                return False
        else:
            # Script shell comme alternative
            marker_shell = Path("scripts/download_marker_models.sh")
            if marker_shell.exists():
                logger.info("Exécution du script download_marker_models.sh")
                result = subprocess.run(
                    ["bash", str(marker_shell)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("✅ Modèles marker téléchargés avec succès (via shell)")
                    return True
                else:
                    logger.error(f"❌ Erreur lors du téléchargement des modèles marker (via shell): {result.stderr}")
                    return False
            else:
                # Créer un script simple pour télécharger les modèles marker
                logger.info("Création d'un script temporaire pour télécharger les modèles marker")
                
                # Créer un répertoire pour les modèles marker
                load_dotenv("config.env")
                marker_dir = Path(os.getenv("MARKER_DIR", "offline_models/marker"))
                ensure_directory_exists(marker_dir)
                
                # Créer un script Python temporaire pour télécharger les modèles
                temp_script = """
import os
import sys
from pathlib import Path

# Configurer l'environnement
marker_dir = Path("{marker_dir}")
marker_dir.mkdir(parents=True, exist_ok=True)
os.environ["MARKER_BASE_PATH"] = str(marker_dir)

try:
    from marker import download_models
    download_models()
    print("✅ Modèles marker téléchargés avec succès")
    sys.exit(0)
except Exception as e:
    print(f"❌ Erreur: {{str(e)}}")
    sys.exit(1)
""".format(marker_dir=marker_dir)
                
                # Écrire le script temporaire
                temp_script_path = Path("./temp_download_marker.py")
                with open(temp_script_path, "w") as f:
                    f.write(temp_script)
                
                # Exécuter le script temporaire
                result = subprocess.run(
                    [sys.executable, str(temp_script_path)],
                    capture_output=True,
                    text=True
                )
                
                # Supprimer le script temporaire
                if temp_script_path.exists():
                    temp_script_path.unlink()
                
                if result.returncode == 0:
                    logger.info("✅ Modèles marker téléchargés avec succès (via script temporaire)")
                    logger.info(result.stdout)
                    return True
                else:
                    logger.error(f"❌ Erreur lors du téléchargement des modèles marker: {result.stderr}")
                    return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement des modèles marker: {str(e)}")
        return False

def download_nltk_models():
    """Télécharge les modèles NLTK nécessaires"""
    logger.info("🔄 Téléchargement des modèles NLTK...")
    
    try:
        # Vérifier si le script NLTK existe
        nltk_script = Path("scripts/download_nltk_models.py")
        if nltk_script.exists():
            # Exécuter le script NLTK
            result = subprocess.run(
                [sys.executable, str(nltk_script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Modèles NLTK téléchargés avec succès")
                return True
            else:
                logger.error(f"❌ Erreur lors du téléchargement des modèles NLTK: {result.stderr}")
                return False
        else:
            # Télécharger manuellement
            logger.info("Script download_nltk_models.py non trouvé, téléchargement manuel...")
            
            try:
                import nltk
                nltk_data_dir = Path('offline_models/nltk_data')
                ensure_directory_exists(nltk_data_dir)
                
                # Définir le chemin de données NLTK
                nltk.data.path.insert(0, str(nltk_data_dir))
                
                # Télécharger les ressources essentielles
                resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
                for resource in resources:
                    nltk.download(resource, download_dir=str(nltk_data_dir))
                
                logger.info("✅ Modèles NLTK téléchargés manuellement avec succès")
                return True
            except ImportError:
                logger.error("❌ La bibliothèque nltk n'est pas installée")
                logger.info("💡 Installation: pip install nltk")
                return False
            except Exception as e:
                logger.error(f"❌ Erreur lors du téléchargement manuel des modèles NLTK: {str(e)}")
                return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement des modèles NLTK: {str(e)}")
        return False

def update_config_files():
    """Met à jour les fichiers de configuration"""
    logger.info("🔄 Mise à jour des fichiers de configuration...")
    
    # Mise à jour de config.env
    config_updates = {
        "USE_OFFLINE_MODELS": "true",
    }
    
    for key, value in config_updates.items():
        update_env_file("config.env", key, value)
    
    # Vérifier que tous les répertoires mentionnés dans config.env existent
    load_dotenv("config.env")
    directories = [
        os.getenv("CACHE_DIR", "offline_models/embeddings_cache"),
        os.getenv("MODELS_DIR", "offline_models/hf"),
        os.getenv("MARKER_DIR", "offline_models/marker"),
        os.getenv("EMBEDDINGS_DIR", "offline_models/embeddings"),
    ]
    
    for directory in directories:
        ensure_directory_exists(directory)
    
    # S'assurer que le répertoire des logs existe
    ensure_directory_exists("logs")
    
    logger.info("✅ Fichiers de configuration mis à jour")
    return True

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du téléchargement de tous les modèles pour utilisation hors ligne")
    
    # 1. Mettre à jour les fichiers de configuration
    update_config_files()
    
    # 2. Télécharger les modèles d'embeddings
    embeddings_success = download_embeddings_models()
    
    # 3. Télécharger les modèles marker
    marker_success = download_marker_models()
    
    # 4. Télécharger les modèles NLTK
    nltk_success = download_nltk_models()
    
    # Résumé
    logger.info("\n📋 Résumé du téléchargement:")
    logger.info(f"  - Modèles d'embeddings: {'✅' if embeddings_success else '❌'}")
    logger.info(f"  - Modèles marker: {'✅' if marker_success else '❌ (optionnel)'}")
    logger.info(f"  - Modèles NLTK: {'✅' if nltk_success else '❌'}")
    
    # Conclusion
    # Considérer les modèles marker comme optionnels
    if embeddings_success and nltk_success:
        if not marker_success:
            logger.warning("⚠️ Les modèles marker n'ont pas été téléchargés, mais ils sont optionnels")
            logger.info("💡 Vous pouvez les télécharger manuellement si nécessaire")
        logger.info("✅ Les modèles essentiels ont été téléchargés avec succès")
        logger.info("🎉 L'application est prête à fonctionner en mode hors ligne!")
        return True
    else:
        logger.warning("⚠️ Certains modèles essentiels n'ont pas été téléchargés correctement")
        logger.info("💡 Consultez les logs pour plus de détails")
        return False

if __name__ == "__main__":
    # Créer le répertoire de logs s'il n'existe pas
    Path("./logs").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1) 