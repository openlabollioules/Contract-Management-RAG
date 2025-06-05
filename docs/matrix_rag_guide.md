# Guide de démarrage - MatrixRAG pour contrats

Ce guide vous explique comment utiliser le système MatrixRAG pour améliorer votre système de RAG (Retrieval-Augmented Generation) pour les contrats en utilisant les matrices G et Doc.

## Présentation

Le système MatrixRAG combine deux approches complémentaires :

1. **Matrice G** : Thésaurus de clauses contractuelles avec identifiants (idG), titres, types, tags et résumés
2. **Matrice Doc** : Extraction des clauses de contrats spécifiques avec référence vers la Matrice G

L'objectif est d'améliorer la précision et la pertinence des réponses en :
- Classifiant automatiquement les clauses (`idG`)
- Filtrant les résultats par type de clause avant la recherche sémantique
- Offrant un fallback vers les définitions canoniques quand aucune clause spécifique n'est trouvée

## Prérequis

- Python 3.8+
- ChromaDB
- Sentence Transformers
- FAISS
- Pandas
- Fichiers CSV/Excel pour les matrices G et Doc

## Installation

1. Assurez-vous que toutes les dépendances sont installées :

```bash
pip install -r requirements.txt
```

2. Préparez vos fichiers de matrices :
   - `matrix_g.csv` : Thésaurus des clauses (colonnes : idG, titre, type, tags, résumé)
   - `matrix_doc.csv` : Clauses extraites (colonnes : idDoc, texte, idG, métadonnées...)

## Indexation des matrices

Pour indexer vos matrices dans ChromaDB :

```bash
python src/utils/matrix_indexer.py \
  --matrix-g data/matrix_g.csv \
  --matrix-doc data/matrix_doc.csv \
  --chroma-path ./chroma_db \
  --collection-g matrix_g \
  --collection-doc matrix_doc
```

## Classification des clauses

Pour classifier automatiquement des chunks de texte en clauses avec leur `idG` correspondant :

```bash
python src/utils/ingest_with_classification.py \
  --matrix-g data/matrix_g.csv \
  --input-chunks data/chunks.json \
  --output-chunks data/chunks_classified.json \
  --index-chroma \
  --chroma-path ./chroma_db \
  --collection-name contract_chunks
```

## Utilisation du retriever amélioré

### Depuis votre code Python

```python
from src.core.matrix_processor import MatrixProcessor
from src.core.enhanced_retriever import create_enhanced_retriever
import chromadb

# Initialiser le processeur de matrices
matrix_processor = MatrixProcessor(
    matrix_g_path="data/matrix_g.csv",
    matrix_doc_path="data/matrix_doc.csv"
)

# Initialiser ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Créer le retriever de base (ChromaDB standard)
base_retriever = create_chromadb_retriever(chroma_client, "contract_chunks")

# Créer le retriever amélioré
enhanced_retriever = create_enhanced_retriever(
    retriever_type="default",  # ou "langchain"
    vector_store=base_retriever,
    matrix_processor=matrix_processor,
    fallback_enabled=True
)

# Utiliser le retriever
query = "Quelles sont les obligations de confidentialité dans le contrat?"
results = enhanced_retriever.retrieve(query, top_k=5)
```

### Avec LangChain

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from src.core.matrix_processor import MatrixProcessor
from src.core.enhanced_retriever import create_enhanced_retriever

# Initialiser les embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialiser le vectorstore LangChain
vectorstore = Chroma(
    collection_name="contract_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialiser le processeur de matrices
matrix_processor = MatrixProcessor(
    matrix_g_path="data/matrix_g.csv",
    matrix_doc_path="data/matrix_doc.csv"
)

# Créer le retriever amélioré pour LangChain
enhanced_retriever = create_enhanced_retriever(
    retriever_type="langchain",
    vector_store=vectorstore,
    matrix_processor=matrix_processor
)

# Utiliser le retriever dans une chaîne LangChain
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

llm = Ollama(model="mistral-large3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=enhanced_retriever
)

response = qa_chain.run("Quelles sont les obligations de confidentialité dans le contrat?")
```

## Démonstration interactive

Pour tester le système avec une interface interactive :

```bash
python src/examples/matrix_rag_demo.py \
  --matrix-g data/matrix_g.csv \
  --matrix-doc data/matrix_doc.csv \
  --chroma-path ./chroma_db \
  --collection-name contract_chunks \
  --interactive
```

Ou exécuter une requête unique :

```bash
python src/examples/matrix_rag_demo.py \
  --matrix-g data/matrix_g.csv \
  --matrix-doc data/matrix_doc.csv \
  --chroma-path ./chroma_db \
  --collection-name contract_chunks \
  --query "Quelles sont les obligations de garantie dans le contrat?"
```

## Fonctionnalités avancées

### Classification des clauses

Pour classifier une clause individuelle :

```bash
python src/main.py \
  --matrix-g data/matrix_g.csv \
  --matrix-doc data/matrix_doc.csv \
  --classify-clause "Le Fournisseur garantit que l'Équipement est conforme aux spécifications techniques..."
```

### Détection d'idG à partir d'une requête

Pour voir quels idG sont détectés dans une question :

```bash
python src/main.py \
  --matrix-g data/matrix_g.csv \
  --matrix-doc data/matrix_doc.csv \
  --detect-idg "Quelles sont les obligations de garantie dans le contrat?"
```

## Structure des fichiers

- `src/core/matrix_processor.py` : Classe principale pour le traitement des matrices
- `src/core/enhanced_retriever.py` : Implémentation du retriever amélioré
- `src/utils/matrix_indexer.py` : Outil pour indexer les matrices
- `src/utils/ingest_with_classification.py` : Pipeline d'ingestion avec classification
- `src/examples/matrix_rag_demo.py` : Démonstration interactive du système
- `src/main.py` : Interface en ligne de commande pour diverses fonctionnalités

## Conseils d'utilisation

1. **Préparation de la Matrice G** :
   - Assurez-vous que chaque clause a un identifiant unique (idG)
   - Utilisez des tags pertinents pour améliorer la détection d'idG
   - Rédigez des résumés concis mais complets pour le fallback

2. **Classification des clauses** :
   - Pour de meilleurs résultats, utilisez des chunks qui correspondent à des clauses entières
   - Ajustez le seuil de similarité (`SIMILARITY_THRESHOLD`) selon vos besoins

3. **Optimisation des performances** :
   - Pour de grands corpus, utilisez un modèle d'embedding plus rapide
   - Ajustez les paramètres de préfiltrage pour équilibrer précision et rappel

4. **Intégration avec des systèmes existants** :
   - Le système est conçu pour être flexible et s'intégrer à différentes architectures
   - Adapter les wrappers selon vos besoins spécifiques 