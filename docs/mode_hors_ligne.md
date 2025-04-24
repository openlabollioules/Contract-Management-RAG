# Fonctionnement du mode hors-ligne

Ce document explique en détail comment configurer et utiliser le mode hors-ligne du système Contract Management RAG, permettant de fonctionner sans connexion Internet.

## Introduction au mode hors-ligne

Le mode hors-ligne du système permet d'exécuter l'ensemble des fonctionnalités sans nécessiter de connexion Internet. Cela est particulièrement utile dans les environnements sécurisés, lors de déplacements, ou dans des situations où la confidentialité des documents est critique.

## Composants du mode hors-ligne

Le mode hors-ligne implique plusieurs composants :

1. **Modèles d'embeddings** : Téléchargés localement pour la vectorisation des textes
2. **Modèles Marker** : Pour l'extraction avancée de texte des PDF
3. **Modèles LLM** via Ollama : Pour la génération de réponses en mode chat
4. **Ressources NLTK** : Pour l'analyse linguistique
5. **Cache d'embeddings** : Pour éviter de recalculer les embeddings fréquemment utilisés

## Installation du mode hors-ligne

### Script automatisé

La méthode la plus simple pour configurer le mode hors-ligne est d'utiliser le script `setup_offline_models.sh` :

```bash
# Rendre le script exécutable
chmod +x setup_offline_models.sh

# Exécuter le script avec les options par défaut
./setup_offline_models.sh
```

Ce script :
1. Crée les répertoires nécessaires pour les modèles
2. Télécharge le modèle d'embeddings configuré
3. Télécharge les modèles Marker pour l'extraction PDF
4. Configure les variables d'environnement
5. Télécharge les ressources NLTK essentielles

### Options du script

Le script `setup_offline_models.sh` accepte plusieurs options :

```bash
# Télécharger tous les modèles (y compris les ressources NLTK complètes)
./setup_offline_models.sh --full

# Télécharger uniquement les modèles essentiels (plus rapide)
./setup_offline_models.sh --minimal

# Vérifier l'état des modèles sans téléchargement
./setup_offline_models.sh --check

# Utiliser l'interface de gestion des modèles
./setup_offline_models.sh --manager
```

### Installation manuelle

Si vous préférez une installation manuelle, suivez ces étapes :

1. Créer les répertoires nécessaires :
```bash
mkdir -p offline_models/marker
mkdir -p offline_models/embeddings
mkdir -p offline_models/embeddings_cache
mkdir -p offline_models/nltk_data
```

2. Télécharger le modèle d'embeddings :
```python
python -c "from document_processing.text_vectorizer import TextVectorizer; TextVectorizer.download_models_for_offline_use()"
```

3. Télécharger les modèles Marker :
```bash
python scripts/download_marker_models.py
```

4. Télécharger les ressources NLTK :
```python
python -c "import nltk; nltk.download('punkt', download_dir='./offline_models/nltk_data'); nltk.download('stopwords', download_dir='./offline_models/nltk_data')"
```

5. Configurer les variables d'environnement dans `config.env` :
```properties
USE_OFFLINE_MODELS=true
OFFLINE_MODELS_DIR=./offline_models
NLTK_DATA=./offline_models/nltk_data
```

## Structure des modèles hors-ligne

Après l'installation, vous aurez une structure de répertoires comme celle-ci :

```
offline_models/
├── embeddings/
│   └── BAAI_bge-m3/           # Modèle d'embeddings
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
├── marker/
│   ├── layout_marker_general  # Modèle de détection de mise en page
│   └── text_marker            # Modèle d'extraction de texte
├── embeddings_cache/
│   └── cache.pkl              # Cache persistant des embeddings
└── nltk_data/
    ├── tokenizers/
    └── corpora/
```

## Configuration du mode hors-ligne

### Variables d'environnement

Le mode hors-ligne est contrôlé par plusieurs variables dans `config.env` :

```properties
# Activation du mode hors-ligne
USE_OFFLINE_MODELS=true

# Répertoire des modèles hors-ligne
OFFLINE_MODELS_DIR=./offline_models

# Utilisation du cache d'embeddings
USE_EMBEDDING_CACHE=true

# Taille maximale du cache d'embeddings
EMBEDDING_CACHE_SIZE=10000

# Chemin vers les données NLTK
NLTK_DATA=./offline_models/nltk_data
```

### Modèles d'embeddings alternatifs

Vous pouvez télécharger différents modèles d'embeddings pour le mode hors-ligne :

```bash
# Modèle léger (plus rapide, moins précis)
EMBEDDING_MODEL=all-MiniLM-L6-v2 ./setup_offline_models.sh

# Modèle multilingue haute qualité
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2 ./setup_offline_models.sh

# Modèle spécialisé pour le français
EMBEDDING_MODEL=camembert-base ./setup_offline_models.sh
```

## Ollama en mode hors-ligne

### Configuration d'Ollama

Ollama doit être configuré séparément pour le mode hors-ligne :

1. Installer Ollama selon les instructions de la [documentation d'installation](installation.md)

2. Télécharger le modèle requis :
```bash
ollama pull mistral-small3.1
```

3. Vérifier que Ollama est en cours d'exécution :
```bash
ollama list
```

### Options pour les modèles LLM

Plusieurs modèles sont disponibles avec Ollama :

- `mistral-small3.1` : Modèle recommandé (bon équilibre performance/qualité)
- `llama2` : Alternative si mistral n'est pas disponible
- `orca-mini` : Option légère pour les systèmes avec ressources limitées

Pour changer de modèle, modifiez la variable `LLM_MODEL` dans `config.env`.

## Cache d'embeddings

### Fonctionnement du cache

Le cache d'embeddings permet d'éviter de recalculer les embeddings pour des textes fréquemment utilisés, améliorant considérablement les performances.

Le système utilise deux niveaux de cache :
1. **Cache en mémoire** : Pour la session en cours
2. **Cache persistant** : Stocké dans `offline_models/embeddings_cache/cache.pkl`

### Gestion du cache

Le cache est géré automatiquement, mais vous pouvez le contrôler avec ces paramètres :

```properties
# Activer/désactiver le cache
USE_EMBEDDING_CACHE=true

# Taille maximale du cache (nombre d'entrées)
EMBEDDING_CACHE_SIZE=10000

# Utiliser un cache persistant
USE_PERSISTENT_CACHE=true

# Fréquence de sauvegarde du cache persistant (secondes)
CACHE_SAVE_INTERVAL=300
```

### Nettoyage du cache

Pour vider le cache d'embeddings :

```bash
# Supprimer uniquement le cache persistant
rm -f offline_models/embeddings_cache/cache.pkl

# OU utiliser le script de maintenance
python scripts/maintenance.py --clear-cache
```

## Utilisation du mode hors-ligne

### Vérification du statut

Pour vérifier que le mode hors-ligne est correctement configuré :

```bash
python scripts/check_offline_status.py
```

Ce script vérifiera :
- La présence de tous les modèles nécessaires
- La configuration des variables d'environnement
- L'état du cache d'embeddings
- La connexion à Ollama

### Commandes en mode hors-ligne

L'utilisation du système en mode hors-ligne est identique au mode normal :

```bash
# Traitement d'un contrat
python main.py chemin/vers/votre/contrat.pdf

# Recherche
python main.py chemin/vers/votre/contrat.pdf "votre requête"

# Mode chat
python main.py chemin/vers/votre/contrat.pdf --chat
```

## Performances en mode hors-ligne

### Comparaison avec le mode en ligne

Le mode hors-ligne peut être légèrement plus lent lors de la première utilisation d'un modèle, mais offre ensuite des performances similaires au mode en ligne grâce au cache d'embeddings.

| Opération | Mode en ligne | Mode hors-ligne |
|-----------|---------------|-----------------|
| Traitement initial | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Recherche | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Chat (1ère question) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Chat (questions suivantes) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Optimisation des performances

Pour optimiser les performances en mode hors-ligne :

1. Assurez-vous que le cache d'embeddings est activé
2. Utilisez des modèles plus légers si nécessaire
3. Augmentez la valeur de `EMBEDDING_CACHE_SIZE` si vous disposez de suffisamment de RAM
4. Réduisez `CHUNK_SIZE` pour traiter des unités de texte plus petites

## Dépannage du mode hors-ligne

### Problèmes courants

| Problème | Cause possible | Solution |
|----------|----------------|----------|
| Modèle d'embeddings non trouvé | Téléchargement incomplet | Relancez `setup_offline_models.sh` |
| Erreur NLTK | Ressources NLTK manquantes | Utilisez `setup_offline_models.sh --full` |
| Performances dégradées | Cache non utilisé | Vérifiez que `USE_EMBEDDING_CACHE=true` |
| Erreur de connexion à Ollama | Service non démarré | Lancez `ollama serve` |
| Modèle LLM non disponible | Modèle non téléchargé | Exécutez `ollama pull <modèle>` |

### Journaux spécifiques

Le système génère des journaux spécifiques au mode hors-ligne dans `Logs/offline_mode.log` qui peuvent aider à diagnostiquer les problèmes.

### Réinitialisation complète

Pour réinitialiser complètement le mode hors-ligne :

```bash
# Supprimer tous les modèles hors-ligne
rm -rf offline_models

# Reconfigurer
./setup_offline_models.sh --full
```

## Scénarios d'utilisation avancés

### Environnements air-gapped

Pour les environnements totalement isolés d'Internet :

1. Téléchargez tous les modèles sur une machine avec accès Internet
2. Créez une archive de tout le répertoire :
```bash
tar -czf contract_management_offline.tar.gz offline_models/
```
3. Transférez cette archive vers l'environnement isolé
4. Extrayez l'archive et configurez le système

### Partage de modèles entre utilisateurs

Pour partager les modèles entre plusieurs utilisateurs d'une organisation :

1. Créez un répertoire partagé pour les modèles
2. Configurez `OFFLINE_MODELS_DIR` pour pointer vers ce répertoire partagé
3. Chaque utilisateur peut avoir son propre cache d'embeddings local

## Conclusion

Le mode hors-ligne offre une flexibilité considérable pour utiliser le système Contract Management RAG dans divers environnements, tout en maintenant toutes les fonctionnalités essentielles. Avec une configuration appropriée, les performances en mode hors-ligne peuvent être pratiquement identiques à celles du mode en ligne, tout en garantissant une confidentialité totale des documents traités. 