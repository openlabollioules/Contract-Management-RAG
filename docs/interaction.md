# Interaction et recherche

Ce document détaille les fonctionnalités d'interaction avec les contrats, notamment la recherche sémantique et le mode chat.

## Vue d'ensemble

Contract Management RAG offre deux modes principaux d'interaction avec les contrats :

1. **Recherche sémantique** - Pour trouver rapidement des passages pertinents
2. **Chat avec les contrats** - Pour poser des questions en langage naturel et obtenir des réponses contextualisées

Ces deux modes sont implémentés dans le module `core/interaction.py` et accessibles via l'interface de ligne de commande `main.py`.

## 1. Recherche sémantique

### Principe de fonctionnement

La recherche sémantique permet de trouver des passages pertinents même si les termes exacts de la requête ne sont pas présents dans le texte. Elle fonctionne selon ce processus :

1. La requête de l'utilisateur est convertie en embedding vectoriel
2. Ce vecteur est comparé à tous les embeddings des chunks stockés dans ChromaDB
3. Les chunks les plus similaires (distance vectorielle la plus faible) sont retournés

### Implémentation

La recherche est implémentée dans la fonction `display_contract_search_results` du module `core/interaction.py` :

```python
def display_contract_search_results(query: str, n_results: int = 5) -> None:
    """
    Search in the contract database

    Args:
        query: Search query
        n_results: Number of results to return
    """
    logger.info(f"\n🔍 Recherche: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)

    # Search
    results = chroma_manager.search(query, n_results=n_results)

    # Display results
    logger.info(f"\n📊 Résultats ({len(results)} trouvés):")
    for i, result in enumerate(results, 1):
        logger.info(f"\n--- Résultat {i} ---")
        logger.info(f"Section: {result['metadata']['section']}")
        logger.info(f"Hiérarchie: {result['metadata']['hierarchy']}")
        logger.info(f"Document: {result['metadata']['document_title']}")
        logger.info(f"Contenu: {result['document'][:200]}...")
        logger.info(f"Distance: {result['distance']:.4f}")
```

### Exemple d'utilisation

```bash
python main.py data/contrat_prestation.pdf "quelles sont les modalités de paiement"
```

### Format des résultats

Chaque résultat contient :
- **Section** : Le numéro ou identifiant de la section concernée
- **Hiérarchie** : La position dans la structure hiérarchique du document
- **Document** : Le titre du document source
- **Contenu** : Le texte pertinent (limité aux 200 premiers caractères dans l'affichage)
- **Distance** : La mesure de similarité (plus elle est proche de 0, plus le résultat est pertinent)

### Optimisations et paramètres

La recherche peut être personnalisée via plusieurs paramètres :

- **Nombre de résultats** : Configurable via le paramètre `n_results`
- **Filtrage par métadonnées** : Possibilité de filtrer par document, section, etc.
- **Seuils de pertinence** : Les résultats avec une distance > 0.6 sont généralement peu pertinents

## 2. Chat avec les contrats

### Principe de fonctionnement

Le mode chat permet d'interagir avec les contrats en langage naturel en suivant ces étapes :

1. La question de l'utilisateur est vectorisée et comparée aux chunks du contrat
2. Les chunks les plus pertinents sont récupérés pour former le contexte
3. Ce contexte et la question sont envoyés à un modèle de langage (LLM)
4. Le LLM génère une réponse en utilisant uniquement les informations du contexte
5. La réponse et les sources sont présentées à l'utilisateur

### Implémentation

Le chat est implémenté dans la fonction `chat_with_contract` du module `core/interaction.py` :

```python
def chat_with_contract(query: str, n_context: int = 3) -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
    """
    logger.info(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = TextVectorizer()
    chroma_manager = VectorDBInterface(embeddings_manager)

    # Search for relevant context
    results = chroma_manager.search(query, n_results=n_context)

    # Prepare context for the prompt
    context = "\n\n".join(
        [
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Chapter: {result['metadata'].get('chapter_title', 'Non spécifié')}\n"
            f"Content: {result['document']}"
            for result in results
        ]
    )

    # Create the prompt with context
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats. 
Voici le contexte pertinent extrait des documents :

{context}

Question de l'utilisateur : {query}

Réponds de manière précise en te basant uniquement sur le contexte fourni. 
Si tu ne trouves pas l'information dans le contexte, dis-le clairement."""

    # Get response from Ollama
    response = llm_chat_call_with_ollama(prompt)
    logger.info("\n🤖 Réponse :")
    logger.info(response)

    # Display sources with metadata
    logger.info("\n📚 Sources :")
    logger.info("=" * 80)
    for i, result in enumerate(results, 1):
        logger.info("\n" + "-" * 40)
        logger.info(f"\nSource {i}/{len(results)}")
        logger.info("-" * 40)

        logger.info(f"Distance: {result['distance']:.4f}")

        # Afficher le contenu
        logger.info(result["metadata"].get("content", result["document"])[:200] + "...")
        logger.info("-" * 40)

    logger.info(f"\n📊 Nombre total de sources: {len(results)}")
```

### Modèle de langage (LLM)

Le système utilise Ollama pour exécuter un modèle de langage localement. Le modèle recommandé est `mistral-small3.1`, mais d'autres modèles peuvent être configurés.

Le module `document_processing/llm_chat.py` gère l'interaction avec Ollama :

```python
def llm_chat_call_with_ollama(prompt: str) -> str:
    """
    Ask a question to Ollama LLM

    Args:
        prompt: The prompt to send to Ollama

    Returns:
        The response from Ollama
    """
    # Configuration
    model = os.getenv("LLM_MODEL", "mistral-small3.1")
    ollama_host = os.getenv("LLM_HOST", "http://localhost:11434")
    
    # Prepare request
    url = f"{ollama_host}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40
        }
    }
    
    # Send request
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logger.error(f"Erreur lors de la communication avec Ollama: {e}")
        return "Désolé, je n'ai pas pu générer de réponse. Vérifiez qu'Ollama est bien en cours d'exécution."
```

### Exemple d'utilisation interactive

```bash
python main.py data/contrat_prestation.pdf --chat

💬 Mode chat activé. Tapez 'exit' pour quitter.

Votre question : Quelles sont les obligations du prestataire?
```

### Personnalisation du chat

Plusieurs aspects du chat peuvent être personnalisés :

- **Nombre de chunks de contexte** : Via le paramètre `n_context` (3 par défaut)
- **Température du LLM** : Contrôle la créativité des réponses (0.1 par défaut pour des réponses factuelles)
- **Modèle utilisé** : Configurable via la variable d'environnement `LLM_MODEL`
- **Prompt template** : Personnalisable pour adapter le comportement du LLM

### Limitations et bonnes pratiques

- **Questions spécifiques** : Les questions précises obtiennent généralement de meilleures réponses
- **Contexte limité** : Le système n'utilise que les chunks les plus pertinents (par défaut 3), ce qui peut limiter la vision globale
- **Sources vérifiables** : Vérifiez toujours les sources présentées pour confirmer la réponse
- **Limite de connaissance** : Le système ne connaît que ce qui est présent dans les contrats analysés

## 3. Fonctionnalités avancées

### Filtrage par métadonnées

Il est possible de filtrer les recherches par métadonnées, par exemple pour se concentrer sur certaines sections ou documents :

```python
# Exemple : Filtrage par section
results = chroma_manager.search(
    query, 
    n_results=n_results,
    filter={"section_number": {"$in": ["3", "3.1", "3.2"]}}
)

# Exemple : Filtrage par document
results = chroma_manager.search(
    query, 
    n_results=n_results,
    filter={"document_title": "Contrat de Prestation de Services"}
)
```

### Contextualisation des réponses

Le système enrichit les réponses avec des informations contextuelles :

- Position hiérarchique exacte dans le document
- Numéros de sections et d'articles
- Références à d'autres clauses liées
- Citations précises du texte original

### Mode d'explication

Une fonctionnalité expérimentale permet de demander des explications sur des clauses complexes :

```bash
python main.py data/contrat_complexe.pdf --chat

Votre question : Explique en termes simples la clause 5.3 sur la limitation de responsabilité

🤖 Réponse :
La clause 5.3 explique essentiellement que l'entreprise ne peut pas être tenue responsable pour plus que le montant que vous avez payé pour ses services...
```

## 4. Extension et personnalisation

### Ajout de nouveaux modèles LLM

Le système peut être étendu pour utiliser d'autres modèles LLM :

1. Modifier le module `document_processing/llm_chat.py` pour ajouter un nouveau connecteur
2. Configurer le nouveau modèle dans `config.env`

### Personnalisation des prompts

Les prompts envoyés au LLM peuvent être personnalisés pour différents cas d'usage :

- Analyse juridique approfondie
- Extraction d'informations spécifiques (dates, montants, parties)
- Comparaison entre différentes clauses

### Intégration d'API externes

Le système peut être étendu pour intégrer des API externes :

- Bases de données juridiques
- Services de traduction
- Services d'analyse supplémentaires 