import os

import nltk


def download_nltk_models():
    """Télécharge et organise les modèles NLTK dans le dossier offline_models"""
    # Définir le chemin des données NLTK
    nltk_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "offline_models", "nltk_data"
    )

    # Créer le répertoire s'il n'existe pas
    os.makedirs(nltk_data_path, exist_ok=True)

    # Définir le chemin des données NLTK
    nltk.data.path = [nltk_data_path]

    # Liste des ressources NLTK nécessaires
    required_resources = ["stopwords", "punkt"]

    # Télécharger chaque ressource
    for resource in required_resources:
        print(f"Téléchargement de la ressource NLTK: {resource}")
        nltk.download(resource, download_dir=nltk_data_path)

    print("✅ Téléchargement des modèles NLTK terminé")
    print(f"Les modèles sont disponibles dans: {nltk_data_path}")


if __name__ == "__main__":
    download_nltk_models()
