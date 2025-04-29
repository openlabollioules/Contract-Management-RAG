# Installation et configuration

Ce guide vous explique pas à pas comment installer et configurer le système Contract Management RAG sur votre environnement.

## Prérequis

Avant de commencer l'installation, assurez-vous que votre système dispose des éléments suivants :

- **Système d'exploitation** : Compatible avec Windows 10/11, macOS 11+ ou Linux (Ubuntu 20.04+ recommandé)
- **Python** : Version 3.8 ou supérieure
- **Espace disque** : Minimum 5 Go pour les modèles et la base de données
- **RAM** : Minimum 8 Go (16 Go recommandé pour de meilleures performances)
- **GPU** : Facultatif mais recommandé pour des performances optimales (compatible CUDA/MPS)
- **Connexion Internet** : Requise pour l'installation initiale et le téléchargement des modèles

## Installation étape par étape

### 1. Clonage du dépôt

```bash
git clone https://github.com/username/Contract-Management-RAG.git
cd Contract-Management-RAG
```

### 2. Création d'un environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.

**Sur Windows :**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Sur macOS/Linux :**
```bash
python -m venv .venv
source .venv/bin/activate
```

> **Note** : Vous remarquerez que votre prompt de terminal change après l'activation, indiquant que vous êtes dans l'environnement virtuel.

### 3. Installation des dépendances

```bash
pip install -r requirements.txt
```

Cette commande installera toutes les dépendances nécessaires, y compris :

- ChromaDB pour la base de données vectorielle
- Les bibliothèques de traitement de PDF
- Les frameworks pour les modèles d'embeddings
- Les utilitaires pour l'OCR et l'analyse de documents

> **Note** : L'installation peut prendre plusieurs minutes en fonction de votre connexion Internet et de la puissance de votre machine.

### 4. Installation d'Ollama (pour le chat avec LLM)

Ollama est utilisé pour exécuter localement les modèles de langage qui génèrent les réponses aux questions.

**Sur macOS :**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Sur Windows :**
Téléchargez et installez depuis le [site officiel d'Ollama](https://ollama.com/download).

**Sur Linux :**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Configuration du modèle Ollama

Une fois Ollama installé, téléchargez le modèle de langage recommandé :

```bash
ollama pull mistral-small3.1
```

> **Note** : Ce téléchargement peut prendre du temps (environ 1-3 Go) en fonction de votre connexion.

### 6. Téléchargement des modèles pour le mode hors-ligne

Le script `setup_offline_models.sh` facilite le téléchargement de tous les modèles nécessaires pour une utilisation sans connexion Internet.

```bash
# Rendre le script exécutable (sur macOS/Linux)
chmod +x setup_offline_models.sh

# Exécuter le script
./setup_offline_models.sh
```

**Sur Windows :**
```bash
setup_offline_models.bat
```

> **Important** : Ce script téléchargera plusieurs modèles (env. 1-2 Go au total) et peut prendre 10-15 minutes selon votre connexion.

## Configuration avancée

### Configuration des variables d'environnement

Le fichier `config.env` contient les principales variables de configuration du système. Voici les paramètres que vous pouvez ajuster :

```properties
# Chemin vers la base de données ChromaDB
CHROMA_DB_DIR=./chroma_db

# Configuration du mode hors-ligne
USE_OFFLINE_MODELS=true
OFFLINE_MODELS_DIR=./offline_models

# Configuration des embeddings
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# Configuration du LLM
LLM_MODEL=mistral-small3.1
LLM_HOST=http://localhost:11434
```

### Configuration de la journalisation

Le fichier `logger_config.env` permet de configurer le niveau de détail et les formats de journalisation.

```properties
# Niveau de journalisation (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Format de journalisation
LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s

# Chemin des logs
LOG_DIR=./Logs
```

## Vérification de l'installation

Pour vérifier que votre installation fonctionne correctement, exécutez la commande suivante :

```bash
python -c "from document_processing.text_vectorizer import TextVectorizer; print('Vectorizer OK' if TextVectorizer().get_embeddings('Test') is not None else 'Vectorizer Error')"
```

Vous devriez voir le message "Vectorizer OK" si tout est correctement configuré.

## Dépannage

### Problèmes courants et solutions

1. **Erreur lors de l'installation des dépendances**
   ```
   Solution : Vérifiez votre version de Python (3.8+ requise) et essayez de mettre à jour pip :
   pip install --upgrade pip
   ```

2. **Ollama n'est pas accessible**
   ```
   Solution : Vérifiez que le service Ollama est bien démarré :
   - Sur macOS/Linux : ollama serve
   - Sur Windows : redémarrez le service Ollama depuis les services Windows
   ```

3. **Modèles hors-ligne non trouvés**
   ```
   Solution : Vérifiez que le script setup_offline_models.sh a été exécuté avec succès et que le chemin dans la variable OFFLINE_MODELS_DIR est correct.
   ```

4. **Problèmes d'OCR avec les PDF**
   ```
   Solution : Vérifiez que Tesseract OCR est installé sur votre système :
   - Sur macOS : brew install tesseract
   - Sur Windows : Téléchargez et installez depuis https://github.com/UB-Mannheim/tesseract/wiki
   - Sur Linux : apt-get install tesseract-ocr
   ```

### Obtenir de l'aide supplémentaire

Si vous rencontrez des problèmes non couverts dans cette section, consultez les autres fichiers de documentation :
- [Dépannage](depannage.md) pour des solutions plus détaillées
- [FAQ](faq.md) pour les questions fréquemment posées

## Prochaines étapes

Maintenant que votre installation est terminée, vous pouvez passer au [Guide d'utilisation](utilisation.md) pour apprendre à utiliser le système. 