#!/bin/bash

# Script pour télécharger tous les modèles nécessaires et configurer l'environnement pour une utilisation hors ligne
# Ce script télécharge les modèles d'embeddings et les modèles Marker pour l'extraction PDF
# Il configure également les variables d'environnement nécessaires

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}      Configuration des modèles hors ligne      ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Créer le répertoire de logs
mkdir -p logs

# Activer l'environnement virtuel si présent
if [ -d ".venv" ]; then
    echo -e "${BLUE}🔄 Activation de l'environnement virtuel...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}✅ Environnement virtuel activé${NC}"
fi

# Analyser les arguments
FULL_DOWNLOAD=false
USE_MANAGER=false
CHECK_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --full) FULL_DOWNLOAD=true ;;
        --manager) USE_MANAGER=true ;;
        --check) CHECK_ONLY=true ;;
        *) echo -e "${YELLOW}⚠️ Argument inconnu: $1${NC}" ;;
    esac
    shift
done

# Installer ou mettre à jour les dépendances
if [ "$CHECK_ONLY" = false ]; then
    echo -e "${BLUE}🔄 Installation/mise à jour des dépendances...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ Certaines dépendances n'ont pas pu être installées. Nous continuerons quand même.${NC}"
    else
        echo -e "${GREEN}✅ Dépendances installées avec succès${NC}"
    fi

    # Installer les bibliothèques nécessaires pour le téléchargement des modèles
    pip install huggingface_hub safetensors -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Impossible d'installer huggingface_hub. Arrêt.${NC}"
        exit 1
    fi
fi

# Créer les répertoires pour les modèles
mkdir -p offline_models/hf
mkdir -p offline_models/marker
mkdir -p offline_models/embeddings
mkdir -p offline_models/embeddings_cache

# Si on utilise le script complet (download_all_models.py)
if [ "$FULL_DOWNLOAD" = true ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}    Téléchargement complet des modèles...     ${NC}"
    echo -e "${BLUE}===============================================${NC}"
    
    echo -e "${BLUE}🔄 Exécution du script download_all_models.py...${NC}"
    python3 scripts/download_all_models.py
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ Le téléchargement complet a rencontré des problèmes${NC}"
    else
        echo -e "${GREEN}✅ Téléchargement complet des modèles réussi${NC}"
    fi
    
    exit 0
fi

# Si on utilise le gestionnaire de modèles offline
if [ "$USE_MANAGER" = true ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}    Gestion des modèles avec le manager...    ${NC}"
    echo -e "${BLUE}===============================================${NC}"
    
    if [ "$CHECK_ONLY" = true ]; then
        echo -e "${BLUE}🔄 Vérification des modèles...${NC}"
        python3 scripts/offline_models_manager.py --check
    else
        echo -e "${BLUE}🔄 Capture et vérification des modèles...${NC}"
        python3 scripts/offline_models_manager.py --capture --check
        
        echo -e "${BLUE}🔄 Tentative de téléchargement des modèles manquants...${NC}"
        python3 scripts/offline_models_manager.py --force-download
    fi
    
    # Quitter après avoir utilisé le gestionnaire
    exit $?
fi

# Sinon, on utilise la méthode de téléchargement directe (par défaut)
if [ "$CHECK_ONLY" = false ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  Téléchargement des modèles d'embeddings...   ${NC}"
    echo -e "${BLUE}===============================================${NC}"

    # Télécharger les modèles d'embeddings
    python3 -c "
    import sys
    sys.path.append('.')
    try:
        from document_processing.text_vectorizer import TextVectorizer
        print('Téléchargement des modèles d\'embeddings...')
        results = TextVectorizer.download_models_for_offline_use()
        success = all(results.values())
        if success:
            print('✅ Tous les modèles d\'embeddings ont été téléchargés avec succès')
        else:
            print('⚠️ Certains modèles d\'embeddings n\'ont pas pu être téléchargés')
            for model, result in results.items():
                print(f'- {model}: {\"✅\" if result else \"❌\"}')
    except Exception as e:
        print(f'❌ Erreur lors du téléchargement des modèles d\'embeddings: {str(e)}')
        sys.exit(1)
    "

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ Problème lors du téléchargement des modèles d'embeddings${NC}"
    else
        echo -e "${GREEN}✅ Modèles d'embeddings téléchargés${NC}"
    fi

    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}    Téléchargement des modèles Marker...       ${NC}"
    echo -e "${BLUE}===============================================${NC}"

    # Télécharger les modèles Marker
    echo -e "${BLUE}🔄 Téléchargement des modèles Marker pour l'extraction PDF...${NC}"
    python3 scripts/download_marker_models.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ Problème lors du téléchargement des modèles Marker${NC}"
        echo -e "${YELLOW}⚠️ Les modèles seront téléchargés automatiquement lors de la première utilisation${NC}"
    else
        echo -e "${GREEN}✅ Modèles Marker téléchargés${NC}"
    fi
fi

# Configurer les variables d'environnement
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}     Configuration du mode hors ligne...       ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Mettre à jour config.env
grep -q "^USE_OFFLINE_MODELS=" config.env
if [ $? -eq 0 ]; then
    sed -i.bak 's/^USE_OFFLINE_MODELS=.*/USE_OFFLINE_MODELS="true"/' config.env
else
    echo 'USE_OFFLINE_MODELS="true"' >> config.env
fi

# Vérifier la configuration
echo -e "${BLUE}🔄 Vérification de la configuration...${NC}"

# Vérifier le contenu du répertoire des modèles
echo -e "${BLUE}📂 Contenu du répertoire des modèles:${NC}"
ls -la offline_models/

# Résumé
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}✅ Configuration terminée${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "📚 Les modèles sont maintenant disponibles hors ligne dans le répertoire 'offline_models'."
echo -e "📝 La variable USE_OFFLINE_MODELS a été définie sur 'true' dans config.env."
echo -e "🚀 Vous pouvez maintenant utiliser l'application sans connexion Internet."
echo -e ""
echo -e "Options supplémentaires:"
echo -e "  ./setup_offline_models.sh --full     : Télécharge tous les modèles (y compris NLTK)"
echo -e "  ./setup_offline_models.sh --manager  : Utilise le gestionnaire de modèles offline"
echo -e "  ./setup_offline_models.sh --check    : Vérifie uniquement l'état des modèles sans téléchargement"
echo -e "${BLUE}===============================================${NC}" 