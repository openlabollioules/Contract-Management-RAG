# Guide d'utilisation

Ce guide détaille les différentes façons d'utiliser le système Contract Management RAG pour traiter, rechercher et interagir avec vos contrats.

## Commandes principales

Le système s'utilise principalement via la ligne de commande, avec le script `main.py` comme point d'entrée. Voici les commandes essentielles :

### 1. Traitement d'un contrat

Pour traiter un nouveau contrat et l'ajouter à la base de données :

```bash
python main.py chemin/vers/votre/contrat.pdf
```

Exemple :
```bash
python main.py data/contrat_prestation.pdf
```

Cette commande va :
1. Extraire le texte du PDF
2. Analyser sa structure et le découper en chunks intelligents
3. Vectoriser ces chunks
4. Les stocker dans la base de données ChromaDB

Sortie attendue :
```
🔄 Début du traitement du document...
📄 Extraction du texte du PDF (avec détection des en-têtes/pieds de page...)
✅ Texte extrait (3421 mots)

🔄 Découpage du texte avec approche hybride (structure + sémantique)...
🔍 Initialisation des embeddings et de ChromaDB...
📦 Préparation des chunks pour ChromaDB...
💾 Ajout des chunks à ChromaDB...
✅ Chunks ajoutés à ChromaDB

Document Metadata:
- Title: Contrat de Prestation de Services
- Author: Unknown
- Pages: Unknown

⏱️ Temps total de traitement: 8.45 secondes
📊 Nombre de chunks créés: 12
📊 Taille moyenne des chunks: 285.1 tokens
```

### 2. Recherche dans un contrat

Pour effectuer une recherche dans les contrats déjà traités :

```bash
python main.py chemin/vers/votre/contrat.pdf "votre requête de recherche"
```

Exemple :
```bash
python main.py data/contrat_prestation.pdf "quelles sont les modalités de résiliation?"
```

Sortie attendue :
```
🔍 Recherche: quelles sont les modalités de résiliation?

📊 Résultats (3 trouvés):

--- Résultat 1 ---
Section: 8.2
Hiérarchie: 8 -> 8.2
Document: Contrat de Prestation de Services
Contenu: Le contrat peut être résilié par l'une ou l'autre des parties en cas de manquement...
Distance: 0.1342

--- Résultat 2 ---
Section: 8
Hiérarchie: 8
Document: Contrat de Prestation de Services
Contenu: Résiliation. Le présent contrat peut être résilié dans les conditions suivantes...
Distance: 0.1573

--- Résultat 3 ---
Section: 8.3
Hiérarchie: 8 -> 8.3
Document: Contrat de Prestation de Services
Contenu: En cas de résiliation anticipée, le Client s'engage à régler les prestations...
Distance: 0.1894
```

### 3. Mode chat avec un contrat

Pour interagir en mode conversation avec un contrat :

```bash
python main.py chemin/vers/votre/contrat.pdf --chat
```

Exemple :
```bash
python main.py data/contrat_prestation.pdf --chat
```

Cette commande entre dans un mode interactif où vous pouvez poser des questions en langage naturel :

```
💬 Mode chat activé. Tapez 'exit' pour quitter.

Votre question : Quelles sont les pénalités en cas de retard de paiement?

🤖 Réponse :
D'après le contrat, tout retard de paiement entraîne l'application d'une pénalité égale à trois fois le taux d'intérêt légal, ainsi que l'application d'une indemnité forfaitaire pour frais de recouvrement de 40€.

📚 Sources :
[Détail des sources utilisées pour la réponse]

Votre question : Quels sont les délais de paiement prévus?
[...]
```

Pour quitter le mode chat, tapez simplement `exit`.

## Exemples d'utilisation avancée

### Traitement de plusieurs contrats

Vous pouvez traiter plusieurs contrats en séquence :

```bash
for fichier in data/*.pdf; do
    python main.py "$fichier"
done
```

### Extraction et recherche dans un workflow

Exemple de workflow complet :

```bash
# Traiter un nouveau contrat
python main.py data/nouveau_contrat.pdf

# Rechercher des informations spécifiques
python main.py data/nouveau_contrat.pdf "obligations du prestataire"

# Entrer en mode chat pour des questions plus complexes
python main.py data/nouveau_contrat.pdf --chat
```

### Optimisation du traitement des PDF numérisés

Pour les PDF numérisés de mauvaise qualité, utilisez l'option d'optimisation OCR :

```bash
python main.py data/contrat_scanne.pdf --ocr-optimize
```

## Astuces et bonnes pratiques

### 1. Rédaction des requêtes

Pour obtenir les meilleurs résultats lors de vos recherches, suivez ces conseils :

- **Soyez spécifique** : "Quelles sont les modalités de paiement?" plutôt que "paiement"
- **Utilisez des termes juridiques** : "clause de non-concurrence" plutôt que "interdiction de concurrence"
- **Posez des questions complètes** : "Quelle est la durée du préavis de résiliation?" plutôt que "préavis"

### 2. Organisation des fichiers

Pour une gestion optimale de vos contrats :

- Créez des sous-dossiers thématiques dans le répertoire `data/`
- Utilisez des noms de fichiers descriptifs : `contrat_prestation_clientX_2023.pdf`
- Conservez une convention de nommage cohérente

### 3. Mode chat efficace

Pour tirer le meilleur parti du mode chat :

- Commencez par des questions générales pour comprendre la structure du contrat
- Précisez progressivement vos questions en fonction des réponses reçues
- Utilisez les numéros d'articles/sections mentionnés dans les réponses pour cibler vos questions
- Si une réponse semble incomplète, reformulez votre question

## Interprétation des résultats

### Comprendre les métriques

- **Distance** : Plus cette valeur est proche de 0, plus le résultat est pertinent (0.3 ou moins indique généralement une bonne correspondance)
- **Hiérarchie** : Indique l'emplacement dans la structure du document (ex: 3 -> 3.2 signifie article 3, section 2)
- **Section** : Numéro de la section ou de l'article concerné

### Fiabilité des réponses en mode chat

Le système s'appuie uniquement sur le contenu des contrats pour générer des réponses. Si une information n'est pas présente dans le document, le système vous l'indiquera explicitement plutôt que d'inventer une réponse.

## Limitations connues

- Le système est optimisé pour les contrats en français et en anglais. D'autres langues peuvent ne pas donner des résultats optimaux.
- Les tableaux complexes et les éléments graphiques ne sont pas toujours correctement interprétés.
- Pour les contrats très volumineux (>100 pages), le traitement peut prendre plusieurs minutes.

## Étapes suivantes

Pour approfondir votre compréhension et personnaliser le système, consultez :
- [Traitement des documents](traitement_documents.md) pour comprendre les détails techniques
- [Configuration et personnalisation](configuration.md) pour adapter le système à vos besoins spécifiques 