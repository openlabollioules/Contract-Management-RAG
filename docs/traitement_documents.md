# Traitement des documents

Ce document explique en détail comment le système traite les documents PDF, depuis l'extraction du texte jusqu'au découpage en chunks sémantiques.

## Vue d'ensemble du processus

Le traitement d'un document dans Contract Management RAG suit ces étapes principales :

1. **Extraction du texte** - Récupération intelligente du contenu textuel du PDF
2. **Analyse de structure** - Identification des sections, articles et hiérarchie du document
3. **Découpage hybride** - Application d'une approche combinant structure et sémantique
4. **Nettoyage et post-traitement** - Restauration du contenu important et suppression du bruit
5. **Enrichissement avec métadonnées** - Ajout d'informations contextuelles à chaque chunk

## 1. Extraction du texte (PDF Extractor)

L'extraction du texte est gérée par le module `pdf_extractor.py` qui combine plusieurs techniques avancées :

### Techniques utilisées

- **Extraction directe** via PyMuPDF (fitz) pour les PDF bien structurés
- **Analyse de mise en page** via Marker pour détecter les en-têtes, pieds de page et zones de texte
- **OCR adaptatif** via Tesseract pour les documents numérisés ou les zones d'image
- **Détection d'orientation** et correction automatique des pages mal orientées

### Suppression des éléments non pertinents

Le système détecte et supprime automatiquement :
- Les en-têtes et pieds de page répétitifs
- Les numéros de page
- Les filigranes et marques de confidentialité
- Les éléments graphiques sans contenu textuel important

### Exemple de traitement

```python
# Extraction avec détection de mise en page
text, document_title = extract_pdf_text("contrat.pdf", 
                                       detect_headers=True, 
                                       remove_watermarks=True)

# Résultat : texte nettoyé sans éléments parasites
```

## 2. Analyse de structure (Contract Splitter)

Le module `contract_splitter.py` analyse la structure du document pour identifier son organisation hiérarchique.

### Identification des composants structurels

Le système recherche et identifie automatiquement :
- **Articles** et sections numérotées (ex: "Article 1", "1.2", "Section 3")
- **Titres** et sous-titres (détection par formatage et contenu)
- **Listes** numérotées ou à puces
- **Annexes** et pièces jointes

### Modèles de reconnaissance

La détection utilise des modèles de reconnaissance adaptés aux formats juridiques courants :
- Patterns numériques (1.1, 1.2, I.a, etc.)
- Patterns textuels ("Article", "Section", "Clause", etc.)
- Analyse de formatage (indentation, capitalisation, etc.)

Ces modèles sont définis dans le fichier `language_patterns.json` et peuvent être personnalisés.

### Création d'une hiérarchie

Le résultat est une représentation hiérarchique du document :

```
Document
├── Article 1: Objet du contrat
│   ├── 1.1: Définition des prestations
│   └── 1.2: Limites du service
├── Article 2: Durée
│   └── 2.1: Renouvellement
└── Article 3: Conditions financières
    ├── 3.1: Prix
    ├── 3.2: Modalités de paiement
    └── 3.3: Retard de paiement
```

## 3. Découpage hybride

Le système utilise une approche hybride combinant structure et sémantique pour un découpage optimal.

### Approche structurelle

Les frontières naturelles du document (articles, sections) sont utilisées comme délimitations primaires :

```python
structure_chunks = splitter.split(text)  # Découpage basé sur la structure
```

### Approche sémantique pour les sections longues

Pour les sections dépassant une certaine taille (800 tokens par défaut), un découpage sémantique supplémentaire est appliqué :

```python
if len(chunk.content.split()) > 800:  # Section trop longue
    sub_chunks = semantic_manager.chunk_text(chunk.content)
```

Le découpage sémantique est réalisé par le module `text_chunker.py` qui :
1. Analyse le flux sémantique du texte
2. Identifie les points de rupture naturels (changements de sujet)
3. Découpe aux endroits optimaux pour préserver le sens

### Préservation des métadonnées

Lors du découpage d'une section en sous-chunks, les métadonnées hiérarchiques sont préservées :

```python
for sub_chunk in sub_chunks:
    sub_chunk.section_number = chunk.section_number
    sub_chunk.hierarchy = chunk.hierarchy
    sub_chunk.document_title = chunk.document_title
    # ... autres métadonnées ...
```

## 4. Nettoyage et post-traitement

Après le découpage, des étapes de post-traitement sont appliquées pour garantir la qualité des chunks.

### Restauration du contenu important

Le module `content_restoration.py` analyse chaque chunk pour détecter et restaurer les informations juridiques critiques qui auraient pu être mal découpées :

- Dates et délais
- Montants et conditions financières
- Noms des parties
- Clauses conditionnelles complètes

### Suppression du contenu redondant

Le système détecte et élimine :
- Les répétitions de titres dans le contenu
- Le texte dupliqué entre chunks adjacents (au-delà du chevauchement intentionnel)
- Les références circulaires

### Optimisation de la taille des chunks

Chaque chunk est optimisé pour atteindre un équilibre entre :
- Cohérence sémantique (compréhension du contenu)
- Taille appropriée pour le traitement vectoriel (généralement entre 200 et 800 tokens)
- Préservation du contexte juridique

## 5. Enrichissement avec métadonnées

Chaque chunk est enrichi avec des métadonnées complètes pour faciliter la recherche et le contexte :

```python
metadata = {
    "section_number": chunk.section_number or "unknown",
    "hierarchy": chunk.hierarchy or ["unknown"],
    "document_title": chunk.document_title or "unknown",
    "parent_section": chunk.parent_section or "unknown",
    "chapter_title": chunk.chapter_title or "unknown",
    "position": position,
    "total_chunks": total_chunks,
    "chunk_size": len(chunk.content.split()),
    "timestamp": time.time(),
}
```

Ces métadonnées permettent :
- De situer précisément chaque chunk dans la structure du document
- D'améliorer la pertinence des recherches
- De reconstruire le contexte lors de la génération de réponses

## Résultat final

Le résultat du traitement est un ensemble de chunks intelligents qui :
1. Préservent la structure juridique du document
2. Maintiennent la cohérence sémantique du contenu
3. Sont optimisés pour la recherche vectorielle
4. Contiennent toutes les métadonnées nécessaires à leur contextualisation

Ces chunks sont ensuite vectorisés et stockés dans ChromaDB pour permettre la recherche sémantique et l'interaction avec le document.

## Personnalisation du traitement

Le comportement du traitement peut être personnalisé via plusieurs paramètres dans `config.env` :

```properties
# Taille des chunks (en tokens approximatifs)
CHUNK_SIZE=800

# Chevauchement entre chunks adjacents
CHUNK_OVERLAP=100

# Seuil de rupture sémantique
BREAKPOINT_THRESHOLD=0.6

# Utilisation de l'OCR pour tous les documents
FORCE_OCR=false

# Détection des en-têtes/pieds de page
DETECT_HEADERS=true
```

Ces paramètres permettent d'adapter le traitement à différents types de documents et besoins spécifiques. 