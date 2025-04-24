# Configuration et personnalisation

Ce document détaille les différentes options de configuration du système Contract Management RAG et explique comment le personnaliser pour l'adapter à vos besoins spécifiques.

## Fichiers de configuration

Le système utilise principalement deux fichiers de configuration :

1. **config.env** - Configuration principale du système
2. **logger_config.env** - Configuration de la journalisation

Ces fichiers utilisent le format standard des fichiers d'environnement avec des paires clé-valeur.

## Configuration principale (config.env)

Le fichier `config.env` contient les paramètres principaux du système :

```properties
# ========== Configuration de la base de données vectorielle ==========
# Chemin vers la base de données ChromaDB
CHROMA_DB_DIR=./chroma_db
# Nom de la collection par défaut
CHROMA_COLLECTION_NAME=contracts

# ========== Configuration des modèles ==========
# Mode hors-ligne (true/false)
USE_OFFLINE_MODELS=true
# Répertoire des modèles hors-ligne
OFFLINE_MODELS_DIR=./offline_models
# Modèle d'embedding à utiliser
EMBEDDING_MODEL=BAAI/bge-m3
# Dimension des vecteurs d'embedding
EMBEDDING_DIMENSION=1024

# ========== Configuration du LLM ==========
# Modèle LLM à utiliser avec Ollama
LLM_MODEL=mistral-small3.1
# URL de l'API Ollama
LLM_HOST=http://localhost:11434
# Température du LLM (0.0-1.0)
LLM_TEMPERATURE=0.1
# Paramètres top_p et top_k
LLM_TOP_P=0.95
LLM_TOP_K=40

# ========== Configuration du traitement des documents ==========
# Taille des chunks (tokens approximatifs)
CHUNK_SIZE=800
# Chevauchement entre chunks adjacents
CHUNK_OVERLAP=100
# Seuil de rupture sémantique (0.0-1.0)
BREAKPOINT_THRESHOLD=0.6
# Utilisation de l'OCR pour tous les documents (true/false)
FORCE_OCR=false
# Détection des en-têtes/pieds de page (true/false)
DETECT_HEADERS=true
# Langue par défaut pour la détection des structures
DEFAULT_LANGUAGE=fr

# ========== Configuration de l'application ==========
# Nombre de résultats à afficher par défaut
DEFAULT_RESULTS_COUNT=5
# Nombre de chunks de contexte pour le chat
CHAT_CONTEXT_SIZE=3
# Mettre en cache les embeddings (true/false)
USE_EMBEDDING_CACHE=true
# Taille maximale du cache d'embeddings (nombre d'entrées)
EMBEDDING_CACHE_SIZE=10000
```

## Configuration de la journalisation (logger_config.env)

Le fichier `logger_config.env` contrôle le comportement du système de journalisation :

```properties
# Niveau de journalisation (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Format de journalisation
LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s

# Répertoire des logs
LOG_DIR=./Logs

# Rotation des fichiers de log
# Taille maximale d'un fichier de log en octets (10 Mo par défaut)
LOG_MAX_SIZE=10485760
# Nombre de backups à conserver
LOG_BACKUP_COUNT=5

# Affichage des logs dans la console (true/false)
LOG_TO_CONSOLE=true
# Couleurs dans la console (true/false)
LOG_COLOR=true
```

## Personnalisation du comportement

### Paramètres clés et leur impact

#### Traitement des documents

| Paramètre | Description | Impact |
|-----------|-------------|--------|
| `CHUNK_SIZE` | Taille des chunks en tokens | Une valeur plus grande conserve plus de contexte mais peut réduire la précision des recherches |
| `CHUNK_OVERLAP` | Chevauchement entre chunks | Augmenter cette valeur améliore la cohérence entre chunks mais augmente la redondance |
| `BREAKPOINT_THRESHOLD` | Seuil de rupture sémantique | Une valeur plus basse crée plus de points de découpage, valeur plus haute préserve des chunks plus grands |
| `DETECT_HEADERS` | Détection des en-têtes | Désactiver si les en-têtes contiennent des informations importantes à conserver |

#### Vectorisation et recherche

| Paramètre | Description | Impact |
|-----------|-------------|--------|
| `EMBEDDING_MODEL` | Modèle d'embedding | Changer de modèle affecte la qualité et la vitesse des recherches |
| `DEFAULT_RESULTS_COUNT` | Nombre de résultats | Augmenter pour avoir plus de contexte, diminuer pour plus de précision |
| `USE_EMBEDDING_CACHE` | Cache des embeddings | Désactiver peut libérer de la mémoire mais ralentit les performances |

#### Génération de réponses

| Paramètre | Description | Impact |
|-----------|-------------|--------|
| `LLM_MODEL` | Modèle de langage | Différents modèles offrent différents équilibres performance/qualité |
| `LLM_TEMPERATURE` | Température | Augmenter rend les réponses plus créatives mais potentiellement moins précises |
| `CHAT_CONTEXT_SIZE` | Taille du contexte | Plus de contexte = réponses plus complètes mais temps de traitement plus long |

### Exemples de personnalisation

#### Configuration pour documents courts (< 10 pages)

```properties
CHUNK_SIZE=500
CHUNK_OVERLAP=50
BREAKPOINT_THRESHOLD=0.7
DEFAULT_RESULTS_COUNT=3
CHAT_CONTEXT_SIZE=2
```

#### Configuration pour documents très longs (> 100 pages)

```properties
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
BREAKPOINT_THRESHOLD=0.5
DEFAULT_RESULTS_COUNT=7
CHAT_CONTEXT_SIZE=5
```

#### Configuration pour performance optimale

```properties
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=20000
LLM_MODEL=mistral-small
FORCE_OCR=false
```

#### Configuration pour précision maximale

```properties
EMBEDDING_MODEL=BAAI/bge-large-en
CHUNK_SIZE=600
CHUNK_OVERLAP=200
LLM_MODEL=mistral-medium
LLM_TEMPERATURE=0.05
CHAT_CONTEXT_SIZE=7
```

## Personnalisation avancée

### Modification des prompts

Les prompts utilisés pour le mode chat peuvent être personnalisés en modifiant le fichier `core/interaction.py` :

```python
# Prompt template original
prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats. 
Voici le contexte pertinent extrait des documents :

{context}

Question de l'utilisateur : {query}

Réponds de manière précise en te basant uniquement sur le contexte fourni. 
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""

# Exemple de prompt personnalisé pour un usage juridique
prompt = f"""Tu es un assistant juridique expert. 
Analyse les extraits de contrat suivants :

{context}

Question juridique : {query}

Réponds avec une analyse juridique complète en citant les numéros de clauses et articles. 
Utilise un langage précis et technique. Si l'information n'est pas disponible, indique-le clairement et suggère 
quelles autres sections du contrat pourraient contenir cette information."""
```

### Personnalisation des modèles de reconnaissance de structure

Les modèles utilisés pour détecter la structure des contrats sont définis dans le fichier `document_processing/language_patterns.json`. Vous pouvez les modifier pour adapter le système à des formats de contrats spécifiques :

```json
{
  "fr": {
    "article_patterns": [
      "Article\\s+([0-9]+|[IVXLCDM]+)",
      "ARTICLE\\s+([0-9]+|[IVXLCDM]+)"
    ],
    "section_patterns": [
      "([0-9]+)\\.([0-9]+)(?:\\.([0-9]+))?",
      "([A-Z])\\.([0-9]+)(?:\\.([0-9]+))?"
    ],
    "title_indicators": [
      "TITRE",
      "CHAPITRE",
      "CLAUSE"
    ]
  },
  "en": {
    "article_patterns": [
      "Article\\s+([0-9]+|[IVXLCDM]+)",
      "Section\\s+([0-9]+|[IVXLCDM]+)"
    ],
    "section_patterns": [
      "([0-9]+)\\.([0-9]+)(?:\\.([0-9]+))?",
      "([A-Za-z])\\.([0-9]+)(?:\\.([0-9]+))?"
    ],
    "title_indicators": [
      "TITLE",
      "CHAPTER",
      "CLAUSE"
    ]
  }
}
```

### Intégration de nouveaux modèles LLM

Pour ajouter un nouveau modèle LLM, vous pouvez étendre le module `document_processing/llm_chat.py` :

```python
def ask_llama3(prompt: str) -> str:
    """
    Version alternative utilisant llama3 via une API différente
    """
    # Configuration
    api_key = os.getenv("LLAMA_API_KEY")
    api_url = os.getenv("LLAMA_API_URL")
    
    # Préparation et envoi de la requête
    # ...
    
    return response
```

Puis modifier la fonction `chat_with_contract` dans `core/interaction.py` pour utiliser cette nouvelle fonction.

## Gestion des langues

### Support multilingue

Le système prend en charge plusieurs langues pour l'analyse de contrats. Pour chaque langue, vous pouvez définir des patterns spécifiques dans `language_patterns.json`.

Pour configurer la langue par défaut, utilisez le paramètre `DEFAULT_LANGUAGE` dans `config.env`.

### Détection automatique de langue

Le système peut détecter automatiquement la langue d'un document en utilisant la bibliothèque `langdetect`. Cette fonctionnalité est implémentée dans le module `document_processing/contract_splitter.py` :

```python
def detect_language(text):
    """
    Détecte la langue du document
    """
    try:
        return langdetect.detect(text[:5000])
    except:
        return os.getenv("DEFAULT_LANGUAGE", "fr")
```

## Sauvegarde et restauration

### Sauvegarde de la base de données vectorielle

Pour sauvegarder votre base de données ChromaDB :

```bash
# Créer une archive de la base de données
tar -czf chroma_backup.tar.gz ./chroma_db
```

### Restauration de la base de données

Pour restaurer une sauvegarde :

```bash
# Arrêter tout processus utilisant la base de données
# Supprimer la base existante
rm -rf ./chroma_db
# Restaurer depuis la sauvegarde
tar -xzf chroma_backup.tar.gz -C ./
```

## Dépannage lié à la configuration

### Problèmes courants

| Problème | Cause possible | Solution |
|----------|----------------|----------|
| Erreur "ModuleNotFoundError" | Dépendance manquante | Vérifiez que toutes les dépendances sont installées avec `pip install -r requirements.txt` |
| Erreur "Connection refused" avec Ollama | Service Ollama non démarré | Démarrez Ollama avec `ollama serve` |
| Erreur "Model not found" | Modèle non téléchargé | Téléchargez le modèle avec `ollama pull mistral-small3.1` |
| Performances lentes | Cache non utilisé ou taille de chunks inadaptée | Activez `USE_EMBEDDING_CACHE` et ajustez `CHUNK_SIZE` |
| Erreur "No module named 'sentence_transformers'" | Environnement virtuel non activé | Activez l'environnement virtuel avant d'exécuter le script |

### Vérification de la configuration

Pour vérifier que votre configuration est correctement chargée :

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv('config.env'); print(os.environ.get('EMBEDDING_MODEL'))"
```

Cette commande devrait afficher le nom du modèle d'embedding configuré.

## Ressources additionnelles

Pour plus d'informations sur la personnalisation avancée, consultez les documents suivants :
- [Mode hors-ligne](mode_hors_ligne.md) pour la configuration détaillée du mode sans connexion
- [Traitement des documents](traitement_documents.md) pour personnaliser le traitement des PDF 