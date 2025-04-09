#!/bin/bash

# Activer l'environnement virtuel si nécessaire
# source venv/bin/activate

# Exécuter le script de téléchargement
python -c "
import sys
from rag.download_models import download_and_save_models

if not download_and_save_models():
    sys.exit(1)
"

# Vérifier si le téléchargement a réussi
if [ $? -eq 0 ]; then
    echo "✅ Tous les modèles ont été téléchargés avec succès!"
    echo "Les modèles sont disponibles dans le dossier offline_models/marker/"
else
    echo "❌ Une erreur s'est produite lors du téléchargement des modèles."
    exit 1
fi 