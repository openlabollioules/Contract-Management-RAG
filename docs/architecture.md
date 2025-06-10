# Architecture du projet

Ce document décrit l'architecture générale du projet Contract Management RAG et la façon dont les différents modules interagissent.

## Vue d'ensemble

Le projet est organisé en modules, chacun ayant une responsabilité spécifique :

- `main.py` : Point d'entrée de l'application qui analyse les arguments de ligne de commande
- `core/` : Contient les modules principaux de l'application
- `document_processing/` : Contient les modules de traitement des documents
- `utils/` : Contient des utilitaires partagés

## Structure des modules

### Core

Le répertoire `core/` contient les modules principaux :

- `contract_processor.py` : Traitement des contrats (extraction, découpage, etc.)
- `interaction.py` : Fonctionnalités d'interaction (chat, recherche)
- `document_manager.py` : Gestion des documents (vérification d'existence, suppression)
- `cli_handler.py` : Gestion des arguments de ligne de commande

### Document Processing

Le répertoire `document_processing/` contient :

- `vectordb_interface.py` : Interface avec ChromaDB
- `text_vectorizer.py` : Gestion des embeddings
- `pdf_extractor.py` : Extraction de texte à partir de PDF
- `llm_chat.py` : Interface avec les modèles de langage

### Utils

Le répertoire `utils/` contient :

- `logger.py` : Configuration du système de journalisation

## Flux de données

1. L'utilisateur exécute `main.py` avec des arguments
2. `cli_handler.py` analyse les arguments et détermine le mode d'opération
3. Selon le mode, les différentes fonctions sont appelées:
   - Mode traitement: `process_contract()` dans `contract_processor.py`
   - Mode chat: `chat_with_contract()` dans `interaction.py`
   - Mode recherche: `display_contract_search_results()` dans `interaction.py`
   - Mode suppression: `delete_document()` dans `document_manager.py`

## Architecture de la base de données

- Nous utilisons ChromaDB comme base de données vectorielle
- Les documents sont stockés sous forme de chunks avec leurs métadonnées
- Chaque chunk est associé à un embedding vectoriel

## Modularité et extensibilité

L'architecture est conçue pour être modulaire et extensible :

- Séparation claire des responsabilités
- Interfaces bien définies entre les modules
- Facile d'ajouter de nouvelles fonctionnalités sans modifier le code existant

## Diagramme d'architecture

```
main.py
  │
  ├── core/
  │    ├── cli_handler.py (Gestion des arguments CLI)
  │    ├── document_manager.py (Gestion des documents)
  │    ├── contract_processor.py (Traitement des contrats)
  │    └── interaction.py (Chat et recherche)
  │
  ├── document_processing/
  │    ├── vectordb_interface.py (Interface avec ChromaDB)
  │    ├── text_vectorizer.py (Gestion des embeddings)
  │    ├── pdf_extractor.py (Extraction de texte)
  │    └── llm_chat.py (Interface avec LLMs)
  │
  └── utils/
       └── logger.py (Configuration de logging)
```

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