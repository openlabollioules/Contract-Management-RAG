# Contract Management RAG

Système de gestion de contrats basé sur l'architecture RAG (Retrieval-Augmented Generation) permettant d'analyser, d'indexer et d'interroger des documents contractuels en langage naturel.

## Fonctionnalités

- **Traitement automatique** des documents PDF contractuels
- **Découpage intelligent** respectant la structure juridique des documents
- **Recherche sémantique** dans les contrats
- **Chat avec les contrats** en langage naturel
- Interface **locale et hors ligne** respectant la confidentialité des documents

## Architecture

Le projet est organisé en modules, chacun ayant une responsabilité spécifique :

- `main.py` : Point d'entrée de l'application
- `core/` : Contient les modules principaux de l'application
- `document_processing/` : Contient les modules de traitement des documents
- `utils/` : Contient des utilitaires partagés

Pour plus de détails, consultez la [documentation d'architecture](docs/architecture.md).

## Installation

### Prérequis

- Python 3.9+
- [Ollama](https://ollama.ai) pour le mode chat

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Configuration

Créez un fichier `config.env` à la racine du projet avec les paramètres suivants :

```env
LLM_MODEL=mistral-small3.1:latest
OLLAMA_URL=http://localhost:11434
USE_OFFLINE_MODELS=true
```

## Utilisation

### Traitement d'un contrat

```bash
python main.py chemin/vers/contrat.pdf
```

### Chat avec les contrats indexés

```bash
python main.py --chat
```

### Recherche dans les contrats

```bash
python main.py --search "modalités de paiement"
```

Pour des instructions détaillées, consultez le [guide d'utilisation](docs/utilisation.md).

## Structure du projet

```
.
├── main.py                     # Point d'entrée principal
├── core/                       # Modules principaux
│   ├── cli_handler.py          # Gestion des arguments de ligne de commande
│   ├── contract_processor.py   # Traitement des contrats
│   ├── document_manager.py     # Gestion des documents
│   └── interaction.py          # Fonctionnalités d'interaction
├── document_processing/        # Traitement des documents
│   ├── llm_chat.py             # Interface avec les modèles de langage
│   ├── pdf_extractor.py        # Extraction de texte à partir de PDF
│   ├── text_vectorizer.py      # Gestion des embeddings
│   └── vectordb_interface.py   # Interface avec ChromaDB
├── utils/                      # Utilitaires
│   └── logger.py               # Configuration du logging
├── docs/                       # Documentation
└── data/                       # Répertoire pour les documents
```

## Documentation

- [Guide d'utilisation](docs/utilisation.md)
- [Architecture du système](docs/architecture.md)
- [Interaction et recherche](docs/interaction.md)
- [Mode hors ligne](docs/mode_hors_ligne.md)

## Licence

Ce projet est sous licence [MIT](LICENSE).

## Contributions

Les contributions sont les bienvenues ! Veuillez consulter notre [guide de contribution](CONTRIBUTING.md) pour plus d'informations. 