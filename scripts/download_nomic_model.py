#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour télécharger le modèle nomic-ai/nomic-embed-text-v2-moe pour utilisation hors ligne.
"""

import os
import sys
import logging
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour importer les modules du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/download_nomic_model.log", mode="w"),
    ],
)
logger = logging.getLogger("download_nomic_model")

def ensure_directory_exists(directory):
    """S'assure qu'un répertoire existe, le crée sinon"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Répertoire vérifié: {directory}")

def download_nomic_model():
    """Télécharge le modèle nomic-ai/nomic-embed-text-v2-moe pour utilisation hors ligne"""
    # Charger les variables d'environnement
    load_dotenv("config.env")
    
    # Obtenir le répertoire de modèles à partir de config.env
    models_dir = Path(os.getenv("MODELS_DIR", "offline_models/hf"))
    ensure_directory_exists(models_dir)
    
    # Modèle à télécharger
    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    
    logger.info(f"🔄 Téléchargement du modèle {model_name}...")
    logger.info(f"Destination: {models_dir}")
    
    try:
        # Définir le dossier cache pour HuggingFace
        os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
        
        # Télécharger le tokenizer
        logger.info(f"Téléchargement du tokenizer pour {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Télécharger le modèle
        logger.info(f"Téléchargement du modèle {model_name}...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Forcer un calcul d'embedding simple pour vérifier que tout fonctionne
        logger.info("Test du modèle téléchargé...")
        test_text = "Ceci est un test pour vérifier que le modèle fonctionne correctement."
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Encoder le texte
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Vérifiez que les embeddings sont bien produits
            if hasattr(outputs, "last_hidden_state"):
                logger.info(f"✅ Test réussi - Dimension des embeddings: {outputs.last_hidden_state.shape}")
            else:
                logger.info(f"✅ Test réussi - Output: {type(outputs)}")
        
        # Sauvegarder le modèle et le tokenizer
        logger.info(f"Sauvegarde du modèle et du tokenizer dans {models_dir}...")
        model_save_path = models_dir / model_name.split("/")[-1]
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained(model_save_path)
        
        logger.info(f"✅ Modèle {model_name} téléchargé et sauvegardé avec succès dans {model_save_path}")
        
        # Ajouter à config.env si nécessaire
        update_embedding_model = input("Voulez-vous définir ce modèle comme modèle d'embedding par défaut dans config.env? (o/n): ").lower()
        if update_embedding_model == 'o' or update_embedding_model == 'oui':
            config_file = Path("config.env")
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_lines = f.readlines()
                
                # Chercher et mettre à jour la variable EMBEDDING_MODEL
                found = False
                for i, line in enumerate(config_lines):
                    if line.strip().startswith("EMBEDDING_MODEL="):
                        config_lines[i] = f'EMBEDDING_MODEL="{model_name}"\n'
                        found = True
                        break
                
                # Ajouter la variable si elle n'existe pas
                if not found:
                    config_lines.append(f'EMBEDDING_MODEL="{model_name}"\n')
                
                # Écrire le fichier mis à jour
                with open(config_file, "w") as f:
                    f.writelines(config_lines)
                
                logger.info(f"✅ Fichier config.env mis à jour avec EMBEDDING_MODEL={model_name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement du modèle {model_name}: {str(e)}")
        return False

if __name__ == "__main__":
    # S'assurer que le répertoire des logs existe
    ensure_directory_exists("logs")
    
    # Télécharger le modèle
    success = download_nomic_model()
    
    if success:
        logger.info("✅ Opération terminée avec succès")
        sys.exit(0)
    else:
        logger.error("❌ Opération terminée avec erreurs")
        sys.exit(1) 