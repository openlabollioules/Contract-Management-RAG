# Vectorisation et embeddings

Ce document explique comment le système convertit le texte des contrats en représentations vectorielles (embeddings) pour permettre la recherche sémantique et l'interaction intelligente.

## Principes fondamentaux

### Qu'est-ce qu'un embedding?

Un embedding est une représentation numérique d'un texte sous forme de vecteur de nombres à virgule flottante. Cette représentation capture le sens et le contexte du texte, permettant de calculer des similarités sémantiques entre différents textes.

Dans Contract Management RAG, chaque chunk de contrat est converti en un vecteur d'embedding, typiquement de dimension 1024, qui encode sa signification sémantique.

### Pourquoi utiliser des embeddings?

Les embeddings permettent :
- De rechercher des informations par similarité conceptuelle, pas seulement par correspondance de mots-clés
- De trouver des informations pertinentes même si elles sont exprimées différemment de la requête
- D'identifier des relations sémantiques entre différentes parties des contrats

## Architecture de vectorisation

### Module principal (TextVectorizer)

Le composant central de la vectorisation est le module `document_processing/text_vectorizer.py` qui gère :
- Le chargement des modèles d'embeddings
- La transformation du texte en vecteurs
- La mise en cache des embeddings pour les performances
- Le support du mode hors-ligne

### Modèle d'embedding utilisé

Le système utilise par défaut le modèle **BAAI/bge-m3** qui offre :
- Une excellente performance pour les textes juridiques et contractuels
- Un bon équilibre entre précision et efficacité
- Une prise en charge multilingue (français, anglais et autres langues)
- Une dimension de vecteur de 1024 qui capture une grande richesse sémantique

### Processus de vectorisation

1. **Prétraitement** : Le texte est normalisé (espaces, ponctuation, casse)
2. **Tokenization** : Découpage en tokens compréhensibles par le modèle
3. **Génération d'embedding** : Le modèle transforme les tokens en vecteur
4. **Normalisation** : Le vecteur est normalisé pour des mesures de distance cohérentes

```python
# Exemple simplifié de vectorisation
def get_embeddings(self, text):
    # Prétraitement
    text = self._preprocess_text(text)
    
    # Vérification du cache
    if text in self.embedding_cache:
        return self.embedding_cache[text]
    
    # Génération de l'embedding
    embedding = self.model.encode(text)
    
    # Normalisation et mise en cache
    normalized_embedding = embedding / np.linalg.norm(embedding)
    self.embedding_cache[text] = normalized_embedding
    
    return normalized_embedding
```

## Base de données vectorielle (ChromaDB)

### Rôle de ChromaDB

ChromaDB est utilisé comme base de données vectorielle pour :
- Stocker efficacement les vecteurs d'embeddings
- Permettre une recherche par similarité rapide
- Conserver les métadonnées associées à chaque chunk
- Persister les données entre les sessions

### Interface avec ChromaDB

Le module `document_processing/vectordb_interface.py` fournit une interface simplifiée pour interagir avec ChromaDB :

```python
# Ajout de documents avec leurs métadonnées
def add_documents(self, documents):
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    # Génération des embeddings
    embeddings = self.embeddings_manager.get_embeddings_batch(texts)
    
    # Création d'IDs uniques
    ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
    
    # Ajout à ChromaDB
    self.collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
```

### Recherche sémantique

La recherche utilise les embeddings pour trouver les chunks les plus pertinents par similarité sémantique :

```python
# Recherche de documents similaires
def search(self, query, n_results=5):
    # Génération de l'embedding de la requête
    query_embedding = self.embeddings_manager.get_embeddings(query)
    
    # Recherche dans ChromaDB par similarité
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Formatage des résultats
    formatted_results = []
    for i in range(len(results["documents"][0])):
        formatted_results.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return formatted_results
```

## Optimisations et fonctionnalités avancées

### Mise en cache des embeddings

Pour améliorer les performances, le système met en cache les embeddings fréquemment utilisés :
- Cache en mémoire pour les requêtes répétées dans une session
- Cache persistant sur disque pour une réutilisation entre sessions

```python
# Configuration du cache
self.embedding_cache = {}  # Cache en mémoire
self.persistent_cache_file = os.path.join(
    self.offline_models_dir, "embeddings_cache", "cache.pkl"
)

# Chargement du cache persistant
if os.path.exists(self.persistent_cache_file):
    with open(self.persistent_cache_file, "rb") as f:
        self.embedding_cache = pickle.load(f)
```

### Mode hors-ligne

Le système prend en charge un mode hors-ligne complet pour fonctionner sans connexion internet :

1. **Téléchargement préalable** des modèles via `setup_offline_models.sh`
2. **Chargement local** des modèles depuis le dossier `offline_models/`
3. **Utilisation du cache** pour éviter de recalculer les embeddings

```python
# Chargement du modèle en mode hors-ligne
if self.use_offline:
    # Chemin vers le modèle local
    model_path = os.path.join(self.offline_models_dir, self.model_name.replace("/", "_"))
    
    # Vérification que le modèle existe
    if not os.path.exists(model_path):
        self.download_models_for_offline_use()
    
    # Chargement du modèle local
    self.model = SentenceTransformer(model_path)
```

### Traitement par lots

Pour les documents volumineux, le système traite les embeddings par lots pour optimiser l'utilisation de la mémoire :

```python
def get_embeddings_batch(self, texts, batch_size=32):
    """Génère des embeddings pour un lot de textes."""
    all_embeddings = []
    
    # Traitement par lots pour économiser la mémoire
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = self.model.encode(batch)
        
        # Normalisation
        for j, emb in enumerate(batch_embeddings):
            batch_embeddings[j] = emb / np.linalg.norm(emb)
            
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

## Métriques de distance

### Similarité cosinus

La recherche utilise la similarité cosinus comme métrique principale :
- Une valeur proche de 0 indique une forte similarité
- Une valeur proche de 2 indique une dissimilarité maximale

Le système interprète généralement :
- 0.0 - 0.2 : Très pertinent
- 0.2 - 0.4 : Pertinent
- 0.4 - 0.6 : Partiellement pertinent
- > 0.6 : Peu pertinent

### Seuils adaptatifs

Le système peut ajuster dynamiquement les seuils de pertinence en fonction :
- Du nombre de résultats souhaités
- De la distribution des scores dans l'ensemble des résultats
- Du type de contrat analysé

## Personnalisation

Le comportement de la vectorisation peut être modifié via le fichier `config.env` :

```properties
# Modèle d'embedding utilisé
EMBEDDING_MODEL=BAAI/bge-m3

# Dimension des vecteurs d'embedding
EMBEDDING_DIMENSION=1024

# Activation du mode hors-ligne
USE_OFFLINE_MODELS=true

# Répertoire des modèles hors-ligne
OFFLINE_MODELS_DIR=./offline_models

# Taille maximale du cache d'embeddings (en nombre d'entrées)
EMBEDDING_CACHE_SIZE=10000
```

## Modèles alternatifs

Le système prend en charge plusieurs modèles d'embeddings alternatifs :

1. **all-MiniLM-L6-v2** - Plus léger, moins précis mais plus rapide
2. **e5-large** - Haute précision mais plus exigeant en ressources
3. **multilingual-e5-large** - Excellent support multilingue

Pour changer de modèle, modifiez simplement la variable `EMBEDDING_MODEL` dans `config.env`. 