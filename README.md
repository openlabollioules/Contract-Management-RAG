# Contract Management RAG

![Contract Management](https://img.shields.io/badge/Contract-Management-blue)
![RAG](https://img.shields.io/badge/RAG-System-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

Un système avancé de Génération Augmentée par Récupération (RAG) pour la gestion et l'analyse de contrats.

## 📋 Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Comment ça marche](#comment-ça-marche)
- [Personnalisation](#personnalisation)
- [Limitations connues](#limitations-connues)
- [Contribuer](#contribuer)
- [Licence](#licence)

## 📝 Présentation

Contract Management RAG est un système intelligent conçu pour extraire, traiter et interroger des contrats sous forme de documents PDF. Il utilise des techniques avancées de traitement du langage naturel et d'apprentissage automatique pour:

- Extraire de manière précise le texte des contrats, en supprimant les en-têtes, pieds de page et signatures
- Découper intelligemment le texte en "chunks" avec une structure hiérarchique
- Créer des embeddings vectoriels pour une recherche sémantique efficace
- Permettre l'interrogation naturelle du contenu des contrats

## 🚀 Fonctionnalités

- **Traitement avancé des PDF** : Extraction précise du texte, correction d'orientation, détection et suppression des éléments non pertinents
- **Découpage intelligent** : Analyse de la structure hiérarchique des documents (sections, sous-sections)
- **Vectorisation sémantique** : Création d'embeddings avec des modèles de pointe (BAAI/bge-m3)
- **Stockage vectoriel** : Utilisation de ChromaDB pour un stockage et une recherche efficaces
- **Chat avec les contrats** : Interface de chat permettant d'interroger les contrats en langage naturel
- **Mode hors-ligne** : Possibilité de fonctionner sans connexion internet avec des modèles locaux

## 🏗️ Architecture

Le système est composé de plusieurs modules clés:

- **pdf_loader.py** : Extraction intelligente du texte des PDFs avec nettoyage avancé
- **intelligent_splitter.py** : Découpage du texte en chunks avec analyse de la structure hiérarchique
- **hierarchical_grouper.py** : Regroupement des chunks selon leur hiérarchie
- **embeddings_manager.py** : Génération et gestion des embeddings vectoriels
- **chroma_manager.py** : Interface avec la base de données vectorielle ChromaDB
- **ollama_chat.py** : Intégration avec Ollama pour les capacités de génération de texte
- **main.py** : Point d'entrée principal avec les fonctions de traitement et d'interrogation

## 📋 Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Environ 2 Go d'espace disque pour les modèles et dépendances
- Facultatif : GPU compatible avec CUDA/MPS pour de meilleures performances

## 🔧 Installation

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/username/Contract-Management-RAG.git
   cd Contract-Management-RAG
   ```

2. Créer et activer un environnement virtuel:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
   ```

3. Installer les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

4. Installer les dépendances pour la lecture des documents:
   ```bash
   pip install opencv-python langchain-experimental
   ```

5. Installer Ollama pour le chat avec les contrats:
   - Suivez les instructions sur [ollama.ai](https://ollama.ai) pour installer Ollama
   - Téléchargez le modèle mistral-small3.1: `ollama pull mistral-small3.1`

6. Télécharger les modèles offline:
   ```bash
   mkdir -p offline_models
   # Les modèles seront téléchargés automatiquement lors de la première utilisation
   ```

## 💻 Utilisation

### Traitement d'un contrat

```bash
python main.py chemin/vers/votre/contrat.pdf
```

### Recherche dans un contrat

```bash
python main.py chemin/vers/votre/contrat.pdf "votre requête de recherche"
```

### Mode chat avec le contrat

```bash
python main.py chemin/vers/votre/contrat.pdf --chat
```

## 📊 Exemples

### Traitement d'un contrat

```bash
python main.py data/exemple_contrat.pdf
```

Sortie:
```
🔄 Début du traitement du document...
📄 Extraction du texte du PDF (avec détection des en-têtes/pieds de page et suppression des références d'images)...
✅ Texte extrait (5432 mots)

🔍 Découpage du texte en chunks intelligents (avec filtrage des sections vides et suppression des titres du contenu)...
🔍 Regroupement hiérarchique des chunks...
🔍 Initialisation des embeddings et de ChromaDB...
📦 Préparation des chunks pour ChromaDB...
💾 Ajout des chunks à ChromaDB...
✅ Chunks ajoutés à ChromaDB

Document Metadata:
- Title: Contrat de Prestation de Services
- Author: Unknown
- Pages: Unknown
```

### Recherche dans un contrat

```bash
python main.py data/exemple_contrat.pdf "quelles sont les conditions de paiement?"
```

Sortie:
```
🔍 Recherche: quelles sont les conditions de paiement?

📊 Résultats (3 trouvés):

--- Résultat 1 ---
Section: 5.2
Hiérarchie: 5 -> 5.2
Document: Contrat de Prestation de Services
Contenu: Les paiements seront effectués dans un délai de 30 jours à compter de la date de réception de la facture...
Distance: 0.1523

--- Résultat 2 ---
Section: 5
Hiérarchie: 5
Document: Contrat de Prestation de Services
Contenu: Conditions de paiement. Le prestataire facturera le client selon les modalités suivantes...
Distance: 0.1845

--- Résultat 3 ---
Section: 5.3
Hiérarchie: 5 -> 5.3
Document: Contrat de Prestation de Services
Contenu: Tout retard de paiement entraînera l'application d'une pénalité de retard égale à trois fois le taux d'intérêt légal...
Distance: 0.2102
```

### Mode chat avec le contrat

```bash
python main.py data/exemple_contrat.pdf --chat

💬 Mode chat activé. Tapez 'exit' pour quitter.

Votre question : Quelles sont les obligations du prestataire?

🤖 Réponse :
Selon le contrat, le prestataire a plusieurs obligations, notamment:
1. Fournir les services décrits dans l'annexe A du contrat avec compétence et diligence
2. Respecter les délais spécifiés dans le calendrier du projet
3. Affecter du personnel qualifié à la réalisation des prestations
4. Signaler immédiatement au client tout problème pouvant affecter la qualité ou les délais de réalisation
5. Garantir la confidentialité des informations du client

📚 Sources :
[Sources listées avec leurs métadonnées]
```

## 🔍 Comment ça marche

1. **Extraction du texte**: Le texte est extrait du PDF avec suppression intelligente des en-têtes, pieds de page, et éléments non pertinents.

2. **Découpage intelligent**: Le texte est analysé pour identifier sa structure hiérarchique (chapitres, sections, sous-sections) et découpé en chunks cohérents.

3. **Regroupement hiérarchique**: Les chunks sont organisés selon leur position dans la hiérarchie du document.

4. **Vectorisation**: Chaque chunk est transformé en vecteur d'embedding représentant son contenu sémantique.

5. **Stockage vectoriel**: Les embeddings et métadonnées sont stockés dans ChromaDB pour une recherche efficace.

6. **Recherche sémantique**: Les requêtes sont transformées en embeddings et comparées aux chunks stockés pour trouver les plus pertinents.

7. **Génération de réponses**: En mode chat, les chunks les plus pertinents sont utilisés comme contexte pour générer une réponse précise.

## ⚙️ Personnalisation

### Modèles d'embeddings

Vous pouvez modifier le modèle d'embeddings utilisé en modifiant la classe `EmbeddingsManager`:

```python
embeddings_manager = EmbeddingsManager(model_name="autre-modele/compatible")
```

### Modèles de LLM

Pour changer le modèle utilisé pour la génération de réponses:

```python
# Dans rag/ollama_chat.py
_ollama_chat = OllamaChat(model="llama3:latest")  # ou autre modèle compatible avec Ollama
```

### Paramètres de découpage

Vous pouvez ajuster les paramètres de découpage du texte en modifiant la classe `IntelligentSplitter`.

## 🚧 Limitations connues

- Performance réduite sur des documents très volumineux (>100 pages)
- Peut avoir des difficultés avec certains formats PDF complexes ou scannés
- Certaines structures de document très atypiques peuvent ne pas être correctement analysées
- Les performances du chat dépendent du modèle LLM utilisé

## 👥 Contribuer

Les contributions sont les bienvenues! Voici comment vous pouvez contribuer:

1. Forker le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committer vos changements (`git commit -m 'Add some amazing feature'`)
4. Pousser vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

---

Développé avec ❤️ pour simplifier l'analyse et la gestion des contrats. 