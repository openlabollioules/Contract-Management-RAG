# Glossaire

Ce glossaire explique les termes techniques utilisés dans la documentation du système Contract Management RAG, afin de faciliter la compréhension pour les utilisateurs non techniques.

## A

### API (Application Programming Interface)
Interface qui permet à deux applications de communiquer entre elles selon un ensemble de règles prédéfinies.

### Annotations
Dans le contexte de l'extraction de PDF, il s'agit de métadonnées ou marquages qui identifient différentes zones d'un document (titres, paragraphes, etc.).

## B

### Base de données vectorielle
Type de base de données spécialement conçue pour stocker et rechercher des vecteurs d'embeddings, optimisée pour les recherches par similarité. Dans notre système, nous utilisons ChromaDB.

### Breakpoint threshold
Seuil utilisé lors du découpage sémantique qui détermine où couper le texte en fonction des changements de thèmes ou de sujets.

## C

### Cache
Système de stockage temporaire qui conserve des données fréquemment utilisées pour accélérer les traitements futurs. Dans notre système, nous utilisons un cache d'embeddings.

### Chunk
Fragment de texte issu du découpage d'un document plus long. Dans notre système, les chunks sont créés de manière à préserver la cohérence sémantique et la structure du document.

### ChromaDB
Base de données vectorielle open-source utilisée par notre système pour stocker et rechercher efficacement les embeddings de texte.

### Contexte (pour LLM)
Ensemble de textes pertinents fournis à un modèle de langage pour l'aider à générer des réponses précises et appropriées à une question.

## D

### Distance (cosinus)
Mesure mathématique de la similarité entre deux vecteurs. Une distance proche de 0 indique une forte similarité, tandis qu'une valeur proche de 2 indique une grande différence.

### Document title
Titre du document extrait automatiquement lors du traitement d'un PDF.

## E

### Embeddings
Représentations vectorielles de textes qui capturent leur signification sémantique dans un espace multidimensionnel, permettant de calculer des similarités entre différents textes.

### Extraction de texte
Processus qui consiste à extraire le contenu textuel d'un document PDF, en éliminant les éléments non pertinents comme les en-têtes et pieds de page.

## H

### Hiérarchie (de document)
Structure organisationnelle d'un document, comprenant des chapitres, sections, sous-sections, etc. Cette structure est préservée dans les métadonnées de nos chunks.

## L

### LLM (Large Language Model)
Modèle de langage de grande taille entraîné sur d'énormes corpus de texte, capable de générer du texte, répondre à des questions et effectuer diverses tâches linguistiques. Dans notre système, nous utilisons des modèles via Ollama.

## M

### Marker
Bibliothèque d'extraction de texte à partir de PDF qui utilise des modèles d'IA pour détecter et extraire intelligemment le contenu.

### Métadonnées
Informations supplémentaires associées à un chunk de texte, comme sa position dans la hiérarchie du document, son numéro de section, etc.

### Mode hors-ligne
Fonctionnalité qui permet au système de fonctionner sans connexion internet en utilisant des modèles préchargés localement.

## N

### NLTK (Natural Language Toolkit)
Bibliothèque Python pour le traitement du langage naturel, utilisée dans notre système pour l'analyse linguistique.

### Normalisation (des vecteurs)
Processus mathématique qui ajuste la longueur d'un vecteur à 1 tout en préservant sa direction, ce qui améliore la cohérence des mesures de similarité.

## O

### OCR (Optical Character Recognition)
Technologie qui convertit des images de texte en texte machine lisible et modifiable. Utilisée pour les PDF numérisés ou contenant des images de texte.

### Ollama
Outil qui permet d'exécuter des LLM localement sur votre ordinateur, utilisé dans notre système pour le mode chat.

## P

### PDF (Portable Document Format)
Format de document électronique développé par Adobe, couramment utilisé pour les contrats et documents officiels.

### PyMuPDF (fitz)
Bibliothèque Python pour le traitement de PDF, utilisée dans notre système pour l'extraction de texte de base.

## R

### RAG (Retrieval Augmented Generation)
Technique qui combine la recherche d'informations pertinentes (retrieval) avec la génération de texte par un LLM, permettant des réponses précises basées sur des documents spécifiques.

### Recherche sémantique
Méthode de recherche qui comprend le sens et le contexte d'une requête plutôt que de simplement faire correspondre des mots-clés.

## S

### Section
Division d'un document juridique, généralement identifiée par un numéro ou un titre.

### Sentence Transformers
Framework Python utilisé pour générer des embeddings de texte de haute qualité.

### Similarité cosinus
Mesure de similitude entre deux vecteurs non nuls dans un espace vectoriel. Elle mesure le cosinus de l'angle entre ces vecteurs, utilisée pour déterminer la pertinence des résultats de recherche.

## T

### Tesseract
Moteur OCR open-source utilisé pour la reconnaissance de texte dans les images et les PDF numérisés.

### Tokenization
Processus de découpage d'un texte en unités plus petites appelées tokens (mots, sous-mots ou caractères), utilisé dans le traitement du langage naturel.

### Top-k / Top-p
Paramètres qui contrôlent la génération de texte par un LLM. Top-k limite les choix aux k tokens les plus probables, tandis que top-p (ou nucleus sampling) sélectionne parmi les tokens dont la probabilité cumulée atteint p.

## V

### Vectorisation
Processus de conversion de texte en vecteurs numériques (embeddings) pouvant être traités mathématiquement.

### VectorDBInterface
Module de notre système qui gère l'interaction avec la base de données vectorielle ChromaDB. 