# Configuration des modèles hors ligne (offline)

## Introduction

Ce document explique comment configurer et utiliser les modèles en mode hors ligne (offline) pour le système Contract Management RAG. Cette fonctionnalité est particulièrement utile dans les environnements avec des restrictions d'accès à Internet ou pour améliorer les performances en évitant les téléchargements répétés des modèles.

## Modèles requis

Le système utilise plusieurs types de modèles:

1. **Modèles d'embeddings** (pour la vectorisation du texte):
   - BAAI/bge-m3 (modèle principal)
   - sentence-transformers/all-MiniLM-L6-v2 (modèle de repli)

2. **Modèles Marker** (pour l'extraction de texte des PDFs):
   - Layout Segmenter (VikParuchuri/marker_layout_segmenter)
   - Texify (VikParuchuri/marker_texify)
   - Autres modèles complémentaires: text_recognition, table_recognition, text_detection, inline_math_detection, ocr_error_detection

3. **Modèles LLM** (via Ollama, pour le chat):
   - mistral-small3.1 (ou autre modèle configuré)

## Structure des répertoires

La structure recommandée pour le stockage des modèles hors ligne est la suivante:

```
offline_models/
├── embeddings/           # Modèles d'embeddings
├── embeddings_cache/     # Cache des embeddings générés
├── hf/                   # Modèles Hugging Face (Transformers)
├── nltk_data/            # Données NLTK (si installées)
└── marker/               # Modèles Marker
    ├── layout/           # Modèle de segmentation de mise en page
    ├── texify/           # Modèle de post-traitement
    ├── text_recognition/ # Modèle de reconnaissance de texte
    ├── text_detection/   # Modèle de détection de texte
    └── ...               # Autres modèles Marker
```

## Configuration automatique avec le script

La méthode la plus simple pour configurer les modèles en mode hors ligne est d'utiliser le script `setup_offline_models.sh` avec différentes options:

### Option de base (recommandée pour la plupart des utilisateurs)

```bash
# Rendre le script exécutable
chmod +x setup_offline_models.sh

# Exécuter le script avec les options de base
./setup_offline_models.sh
```

Cette commande télécharge les modèles d'embeddings et les modèles Marker essentiels (layout et texify).

### Options avancées

Le script supporte plusieurs options pour des besoins spécifiques:

```bash
# Télécharger tous les modèles (y compris NLTK)
./setup_offline_models.sh --full

# Utiliser le gestionnaire de modèles offline (pour capturer des modèles déjà téléchargés)
./setup_offline_models.sh --manager

# Vérifier uniquement l'état des modèles sans téléchargement
./setup_offline_models.sh --check
```

## Scripts disponibles

Le système fournit plusieurs scripts pour gérer les modèles hors ligne:

### 1. setup_offline_models.sh

Script principal qui orchestre le téléchargement et la configuration des modèles:
- Configure les variables d'environnement
- Télécharge les modèles d'embeddings et Marker
- Offre des options pour utiliser les autres scripts

### 2. download_marker_models.py

Script spécialisé pour télécharger les modèles Marker depuis Hugging Face:
```bash
python scripts/download_marker_models.py
```

Ce script télécharge les modèles layout et texify qui sont essentiels pour l'extraction de texte des PDFs.

### 3. download_all_models.py

Script complet qui télécharge tous les modèles nécessaires, y compris NLTK:
```bash
python scripts/download_all_models.py
```

Ce script est utile pour une configuration hors ligne complète, incluant:
- Tous les modèles d'embeddings
- Tous les modèles Marker
- Les ressources NLTK
- Configuration des variables d'environnement

### 4. offline_models_manager.py

Un outil avancé pour gérer les modèles Marker/Surya:
```bash
# Vérifier l'état des modèles locaux
python scripts/offline_models_manager.py --check

# Capturer les modèles du cache vers offline_models
python scripts/offline_models_manager.py --capture

# Forcer le téléchargement des modèles manquants
python scripts/offline_models_manager.py --force-download

# Spécifier des modèles particuliers
python scripts/offline_models_manager.py --models layout,texify --check
```

Ce script est particulièrement utile pour:
- Capturer des modèles déjà téléchargés par Surya
- Vérifier l'intégrité des modèles
- Gérer spécifiquement les modèles avancés comme text_recognition, text_detection, etc.

## Configuration manuelle

Si vous préférez configurer manuellement les modèles:

### 1. Créer les répertoires nécessaires

```bash
mkdir -p offline_models/marker/layout
mkdir -p offline_models/marker/texify
mkdir -p offline_models/embeddings
mkdir -p offline_models/embeddings_cache
mkdir -p offline_models/hf
```

### 2. Télécharger les modèles d'embeddings

```bash
# Depuis Python
python -c "from document_processing.text_vectorizer import TextVectorizer; TextVectorizer.download_models_for_offline_use()"

# Ou directement avec Hugging Face Hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='offline_models/embeddings/BAAI_bge-m3', local_dir_use_symlinks=False)"
```

### 3. Télécharger les modèles Marker

```bash
# Avec le script dédié
python scripts/download_marker_models.py

# Ou directement avec Hugging Face Hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('VikParuchuri/marker_layout_segmenter', local_dir='offline_models/marker/layout', local_dir_use_symlinks=False)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('VikParuchuri/marker_texify', local_dir='offline_models/marker/texify', local_dir_use_symlinks=False)"
```

### 4. Configurer les variables d'environnement

Modifiez le fichier `config.env` pour activer le mode hors ligne:

```
USE_OFFLINE_MODELS="true"
MARKER_DIR="offline_models/marker"
EMBEDDINGS_DIR="offline_models/embeddings"
CACHE_DIR="offline_models/embeddings_cache"
MODELS_DIR="offline_models/hf"
```

## Variables d'environnement

Les variables d'environnement suivantes contrôlent le comportement du mode hors ligne:

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| USE_OFFLINE_MODELS | Active le mode hors ligne | "true" |
| CACHE_DIR | Répertoire pour le cache des embeddings | "offline_models/embeddings_cache" |
| MODELS_DIR | Répertoire de base pour les modèles HF | "offline_models/hf" |
| MARKER_DIR | Répertoire pour les modèles Marker | "offline_models/marker" |
| EMBEDDINGS_DIR | Répertoire pour les modèles d'embeddings | "offline_models/embeddings" |

## Dépannage

### Problèmes courants

#### 1. Erreur "No module named 'huggingface_hub'"

**Solution**: Installez la bibliothèque Hugging Face Hub:
```bash
pip install huggingface_hub
```

#### 2. Erreur lors du chargement des modèles d'embeddings

**Symptôme**: 
```
Error: Impossible de charger le modèle BAAI/bge-m3 en mode hors ligne
```

**Solutions**:
- Vérifiez que le modèle est correctement téléchargé dans le répertoire `offline_models/embeddings/BAAI_bge-m3`
- Exécutez à nouveau le script de téléchargement: `./setup_offline_models.sh`
- Vérifiez que la variable `USE_OFFLINE_MODELS` est définie sur "true" dans `config.env`

#### 3. Erreur lors de l'extraction du texte des PDFs

**Symptôme**:
```
Error: Échec de l'extraction avec Marker
```

**Solutions**:
- Vérifiez que les modèles Marker sont correctement téléchargés dans `offline_models/marker/layout` et `offline_models/marker/texify`
- Exécutez la commande: `python scripts/download_marker_models.py`
- Utilisez le gestionnaire avancé: `python scripts/offline_models_manager.py --check --force-download`
- Le système utilisera automatiquement une méthode d'extraction plus simple comme repli

#### 4. Les modèles sont toujours téléchargés malgré le mode hors ligne

**Solutions**:
- Assurez-vous que toutes les variables d'environnement sont correctement définies:
  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  ```
- Redémarrez l'application après avoir modifié les variables d'environnement

#### 5. Modèles avancés manquants (text_recognition, etc.)

**Solutions**:
- Utilisez l'option `--full` pour télécharger tous les modèles: `./setup_offline_models.sh --full`
- Ou utilisez le gestionnaire avancé: `python scripts/offline_models_manager.py --force-download`
- Certains modèles avancés sont téléchargés lors de la première utilisation puis capturés via le gestionnaire

## Utilisations avancées

### Utiliser des modèles d'embeddings alternatifs

Vous pouvez utiliser d'autres modèles d'embeddings en les téléchargeant et en modifiant la configuration:

1. Téléchargez le modèle alternatif:
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='offline_models/embeddings/all-MiniLM-L6-v2', local_dir_use_symlinks=False)"
   ```

2. Modifiez `config.env` pour utiliser ce modèle:
   ```
   EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   ```

### Précharger le cache d'embeddings

Pour améliorer les performances, vous pouvez précharger le cache d'embeddings:

```bash
python -c "from document_processing.text_vectorizer import TextVectorizer; vectorizer = TextVectorizer(); vectorizer.get_embeddings(['Exemple de texte à vectoriser', 'Autre exemple'])"
```

### Capturer des modèles depuis le cache Surya

Si vous avez déjà utilisé Surya (qui est intégré à Marker), vous pouvez capturer les modèles téléchargés:

```bash
python scripts/offline_models_manager.py --capture --check
```

Cela copiera tous les modèles du cache Surya (généralement dans `~/Library/Caches/datalab/models`) vers votre répertoire `offline_models`.

## Conclusion

Le mode hors ligne permet d'utiliser le système Contract Management RAG dans des environnements à connectivité limitée ou restreinte. Une fois configuré, le système fonctionnera sans nécessiter d'accès Internet pour le téléchargement des modèles.

Pour toute assistance supplémentaire ou problème non résolu, consultez les logs dans le répertoire `./logs/` ou ouvrez une issue sur le dépôt GitHub du projet. 