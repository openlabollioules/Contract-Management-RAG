# Guide d'utilisation

Ce document détaille les différentes façons d'utiliser l'application Contract Management RAG.

## Prérequis

- Python 3.9+
- Dépendances installées : `pip install -r requirements.txt`
- Ollama installé et lancé (pour le mode chat)

## Modes d'utilisation

L'application offre plusieurs modes d'utilisation :

1. **Traitement de documents** - Analyse et indexation des contrats
2. **Chat interactif** - Dialogue en langage naturel avec les contrats
3. **Recherche** - Recherche sémantique dans les contrats
4. **Suppression** - Suppression de documents de la base de données

## Ligne de commande

### Syntaxe générale

```bash
python main.py <fichiers_contrats> [options]
```

### Options disponibles

- `--chat` : Mode chat interactif
- `--search <requête>` : Recherche dans les contrats
- `--force` : Force le retraitement même si les documents existent déjà
- `--delete` : Supprime les documents spécifiés
- `--debug` : Active les logs détaillés

### Exemples d'utilisation

#### Traitement d'un contrat

```bash
python main.py data/contrat.pdf
```

#### Traitement de plusieurs contrats

```bash
python main.py data/contrat1.pdf data/contrat2.pdf
```

#### Forcer le retraitement

```bash
python main.py data/contrat.pdf --force
```

#### Mode chat avec des contrats spécifiques

```bash
python main.py data/contrat1.pdf data/contrat2.pdf --chat
```

#### Mode chat avec tous les contrats disponibles

```bash
python main.py --chat
```

#### Recherche dans les contrats

```bash
python main.py data/contrat.pdf --search "modalités de paiement"
```

#### Suppression de documents

```bash
python main.py data/contrat_obsolete.pdf --delete
```

## Flux de travail typique

1. **Indexer des contrats** : Ajoutez des contrats à la base de données
   ```bash
   python main.py data/contrat1.pdf data/contrat2.pdf
   ```

2. **Interagir en mode chat** : Posez des questions sur les contrats indexés
   ```bash
   python main.py --chat
   ```

3. **Rechercher des informations** : Effectuez des recherches spécifiques
   ```bash
   python main.py --search "clause de confidentialité"
   ```

## Gestion des erreurs

- Si un document existe déjà, utilisez `--force` pour le réindexer
- Si vous rencontrez des problèmes, utilisez `--debug` pour des logs détaillés
- Pour nettoyer la base de données, vous pouvez supprimer les documents avec `--delete`

## Architecture interne

L'application est organisée de manière modulaire :

- `main.py` : Point d'entrée principal
- `core/` : Modules principaux (traitement, interaction, gestion)
- `document_processing/` : Traitement des documents (extraction, vectorisation)
- `utils/` : Utilitaires (logging, etc.)

Pour plus de détails sur l'architecture, consultez [architecture.md](architecture.md).

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

### 2. Traitement de plusieurs contrats

Vous pouvez traiter plusieurs contrats en une seule commande :

```bash
python main.py contrat1.pdf contrat2.pdf contrat3.pdf
```

Le système traitera chaque contrat séquentiellement. Si un document existe déjà dans la base de données, le programme s'arrêtera avec un message d'erreur :

```
❌ ERREUR : Les documents suivants existent déjà dans la base de données :
   - contrat1.pdf
   - contrat2.pdf

Pour forcer le retraitement, utilisez l'option --force
Pour supprimer ces documents, utilisez l'option --delete
```

Pour forcer le retraitement de documents déjà existants, utilisez l'option `--force` :

```bash
python main.py contrat1.pdf contrat2.pdf --force
```

### 3. Suppression de documents

Pour supprimer des documents de la base de données :

```bash
python main.py contrat1.pdf contrat2.pdf --delete
```

Cette commande supprimera tous les chunks associés aux documents spécifiés.

### 4. Recherche dans les contrats

Pour effectuer une recherche dans les contrats déjà traités :

```bash
python main.py contrat1.pdf contrat2.pdf --search "votre requête de recherche"
```

Exemple :
```bash
python main.py data/contrat_prestation.pdf --search "quelles sont les modalités de résiliation?"
```

Cette commande traitera d'abord les documents spécifiés s'ils ne sont pas déjà dans la base de données, puis effectuera la recherche. Si des documents existent déjà, le programme s'arrêtera avec un message d'erreur, sauf si l'option `--force` est utilisée.

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

### 5. Mode chat avec les contrats

Pour interagir en mode conversation avec des contrats :

```bash
python main.py contrat1.pdf contrat2.pdf --chat
```

Exemple :
```bash
python main.py data/contrat_prestation.pdf data/contrat_confidentialité.pdf --chat
```

Cette commande traite d'abord les documents spécifiés s'ils ne sont pas déjà dans la base de données, puis entre dans un mode interactif où vous pouvez poser des questions en langage naturel pour interroger plusieurs contrats simultanément. Si des documents existent déjà, le programme s'arrêtera avec un message d'erreur, sauf si l'option `--force` est utilisée.

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

## Récapitulatif des options de ligne de commande

```
Usage: python main.py <contract_file1> [contract_file2 ...] [--chat|--search <search_query>] [--force] [--delete]

Options:
  --chat                 Mode chat interactif avec les contrats
  --search <query>       Recherche dans les contrats
  --force                Force le retraitement des documents même s'ils existent déjà
  --delete               Supprime les documents spécifiés de la base de données
```

## Persistance des données

Le système utilise ChromaDB comme base de données vectorielle persistante. Cela signifie que :

1. Les documents traités sont conservés entre les sessions
2. Vous n'avez pas besoin de retraiter les mêmes documents à chaque utilisation
3. Les modifications (ajouts/suppressions) sont permanentes et stockées sur disque

La base de données est stockée par défaut dans le dossier `chroma_db` à la racine du projet.

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

### 4. Gestion de la base de données

Pour maintenir une base de données optimale :

- Supprimez les documents obsolètes avec l'option `--delete` avant d'ajouter une version mise à jour
- Utilisez l'option `--force` pour remplacer un document qui a été modifié
- Évitez d'ajouter des documents avec des noms identiques pour prévenir les erreurs
- Pour une réinitialisation complète, supprimez manuellement le dossier `chroma_db`

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