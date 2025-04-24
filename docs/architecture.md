# Architecture du système

## Vue d'ensemble

Le système Contract Management RAG est conçu selon une architecture modulaire qui permet un traitement efficace des documents contractuels, depuis l'extraction du texte jusqu'à l'interaction via des requêtes en langage naturel. L'architecture est organisée autour de plusieurs composants clés qui travaillent ensemble pour offrir une expérience complète.

## Schéma général

```
┌───────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│                   │     │                    │     │                   │
│  Entrée document  │────▶│ Traitement du texte │────▶│ Vectorisation     │
│  (PDF)            │     │                    │     │                   │
│                   │     └────────────────────┘     └─────────┬─────────┘
└───────────────────┘                                          │
                                                              │
┌───────────────────┐     ┌────────────────────┐     ┌─────────▼─────────┐
│                   │     │                    │     │                   │
│  Interaction      │◀────│ Génération de      │◀────│ Base de données   │
│  (Chat/Recherche) │     │ réponses (LLM)     │     │ vectorielle       │
│                   │     │                    │     │                   │
└───────────────────┘     └────────────────────┘     └───────────────────┘
```

## Composants principaux

### 1. Module principal (`main.py`)

Le point d'entrée du système qui coordonne l'ensemble des opérations et gère les différentes commandes :
- Traitement d'un nouveau contrat
- Recherche dans les contrats existants
- Mode chat interactif

### 2. Traitement des documents

#### PDF Extractor (`document_processing/pdf_extractor.py`)
- Extraction intelligente du texte des PDFs
- Détection et suppression des en-têtes/pieds de page
- Correction de l'orientation des pages
- Optimisation OCR pour les documents numérisés

#### Contract Splitter (`document_processing/contract_splitter.py`)
- Analyse de la structure hiérarchique du document (sections, sous-sections)
- Découpage préliminaire basé sur la structure juridique

#### Text Chunker (`document_processing/text_chunker.py`)
- Découpage sémantique des grandes sections
- Maintien de la cohérence du contenu
- Optimisation de la taille des chunks pour l'indexation vectorielle

### 3. Vectorisation et stockage

#### Text Vectorizer (`document_processing/text_vectorizer.py`)
- Génération d'embeddings vectoriels pour chaque chunk
- Utilisation de modèles avancés (BAAI/bge-m3)
- Support du mode hors-ligne avec cache local

#### VectorDB Interface (`document_processing/vectordb_interface.py`)
- Interface avec ChromaDB pour le stockage des vecteurs
- Indexation et recherche sémantique
- Persistance des données pour réutilisation

### 4. Core (Logique métier)

#### Contract Processor (`core/contract_processor.py`)
- Orchestration du processus complet de traitement
- Application d'une approche hybride (structure + sémantique)
- Préservation des métadonnées hiérarchiques

#### Interaction (`core/interaction.py`)
- Gestion des requêtes de recherche
- Interfaçage avec les LLM pour le mode chat
- Formatage des résultats pour une meilleure lisibilité

#### Content Restoration (`core/content_restoration.py`)
- Post-traitement pour restaurer le contenu important
- Détection et traitement des informations juridiques critiques

### 5. Utilitaires

#### Logger (`utils/logger.py`)
- Configuration centralisée de la journalisation
- Formatage des messages pour une meilleure lisibilité
- Support des différents niveaux de verbosité

## Flux de données

1. **Entrée** : Un document PDF est fourni au système via `main.py`
2. **Extraction** : Le texte est extrait avec `pdf_extractor.py`
3. **Découpage** : Le texte est analysé et découpé par `contract_splitter.py` et `text_chunker.py`
4. **Vectorisation** : Chaque chunk est transformé en vecteur par `text_vectorizer.py`
5. **Stockage** : Les vecteurs et métadonnées sont stockés dans ChromaDB via `vectordb_interface.py`
6. **Recherche/Chat** : L'utilisateur interagit avec le système via `main.py` et `interaction.py`
7. **Génération** : Les réponses sont générées grâce à `llm_chat.py` en utilisant le contexte pertinent

## Interactions entre composants

- **Contract Processor ↔ PDF Extractor** : Initialise l'extraction du texte
- **Contract Processor ↔ Contract Splitter** : Coordonne le découpage structurel
- **Contract Processor ↔ Text Chunker** : Applique le découpage sémantique aux grandes sections
- **Contract Processor ↔ Text Vectorizer** : Gère la vectorisation des chunks
- **Contract Processor ↔ VectorDB Interface** : Coordonne le stockage des vecteurs
- **Interaction ↔ VectorDB Interface** : Récupère les chunks pertinents lors des recherches
- **Interaction ↔ LLM Chat** : Communique avec le LLM pour générer des réponses

## Technologies utilisées

- **Python** : Langage principal du système
- **ChromaDB** : Base de données vectorielle pour le stockage et la recherche
- **Sentence Transformers** : Framework pour les modèles d'embeddings
- **Ollama** : Interface avec des LLM locaux pour la génération de réponses
- **PyMuPDF (Fitz)** : Traitement avancé des PDF
- **Marker** : Analyse de mise en page et extraction de texte
- **ONNX Runtime** : Exécution optimisée des modèles d'IA

Cette architecture modulaire permet une maintenance facile, une extensibilité et une adaptation à différents cas d'utilisation de gestion de contrats. 