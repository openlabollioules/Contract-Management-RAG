import os
import torch
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
import pickle
import shutil

def download_and_save_models():
    # Créer le dossier pour les modèles
    base_path = "offline_models/marker"
    os.makedirs(base_path, exist_ok=True)

    # Configuration pour le téléchargement
    config = {
        "output_format": "markdown",
        "use_llm": True,
        "llm_service": "marker.services.ollama.OllamaService",
        "ollama_model": "command-a:latest",
        "ocr_engine": "tesseract",
        "ocr_language": "fra+eng",
        "table_structure": True,
        "preserve_layout": True,
        "extract_images": True,
        "clean_text": True,
        "remove_headers_footers": True,
        "detect_columns": True,
        "max_workers": 1,
        "batch_size": 1
    }

    try:
        print("Début du téléchargement des modèles...")
        
        # Créer le parser de configuration
        config_parser = ConfigParser(config)
        
        # Créer le dictionnaire des modèles
        model_dict = create_model_dict()
        
        # Pour chaque modèle dans le dictionnaire
        for model_name, model in model_dict.items():
            model_dir = os.path.join(base_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"Téléchargement du modèle {model_name}...")
            
            # Sauvegarder le modèle et ses fichiers associés
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(model_dir)
            else:
                # Si le modèle n'a pas de méthode save_pretrained, on essaie de copier ses fichiers
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"Modèle {model_name} sauvegardé dans {model_dir}")
        
        print("Tous les modèles ont été téléchargés et sauvegardés avec succès!")
        return True
        
    except Exception as e:
        print(f"Erreur lors du téléchargement des modèles: {str(e)}")
        return False

if __name__ == "__main__":
    download_and_save_models() 