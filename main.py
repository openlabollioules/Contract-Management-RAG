import re
import sys
import time
from typing import List

from rag.chroma_manager import ChromaDBManager
from rag.embeddings_manager import EmbeddingsManager
from rag.hierarchical_grouper import HierarchicalGrouper
from rag.intelligent_splitter import Chunk, IntelligentSplitter
from rag.pdf_loader import extract_text_contract
from rag.semantic_chunker import SemanticChunkManager


def display_chunks_details(chunks: List[Chunk]) -> None:
    """
    Affiche le contenu détaillé de chaque chunk avec ses métadonnées

    Args:
        chunks: Liste des chunks à afficher
    """
    print("\n📋 Détails des chunks:")
    print("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}/{len(chunks)}")
        print("-" * 40)
        print(f"Section: {chunk.section_number}")
        print(f"Hiérarchie: {' -> '.join(chunk.hierarchy)}")
        print(f"Document: {chunk.document_title}")
        print(f"Chapitre: {chunk.chapter_title}")
        print(f"Section parente: {chunk.parent_section}")
        print(
            f"Position: {getattr(chunk, 'position', 'N/A')}/{getattr(chunk, 'total_chunks', 'N/A')}"
        )
        print(f"Taille (mots): {len(chunk.content.split())}")
        print("\nContenu:")
        print(chunk.content)
        print("-" * 40)


def display_removed_content(original_text: str, chunks: List[Chunk]) -> None:
    """
    Affiche le contenu qui a été supprimé lors du découpage, en limitant drastiquement ce qui est considéré comme un titre

    Args:
        original_text: Texte original du document
        chunks: Liste des chunks après découpage
    """
    print("\n🗑️ Contenu supprimé lors du découpage:")
    print("=" * 80)

    # Créer une version normalisée du texte original avec contexte
    original_paragraphs = []
    current_paragraph = []

    for line in original_text.split("\n"):
        line = line.strip()
        if not line:
            if current_paragraph:
                original_paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line)

    if current_paragraph:
        original_paragraphs.append(" ".join(current_paragraph))

    # Créer une version normalisée du texte des chunks
    chunk_texts = []
    for chunk in chunks:
        chunk_paragraphs = []
        current_paragraph = []

        for line in chunk.content.split("\n"):
            line = line.strip()
            if not line:
                if current_paragraph:
                    chunk_paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)

        if current_paragraph:
            chunk_paragraphs.append(" ".join(current_paragraph))

        chunk_texts.extend(chunk_paragraphs)

    # Liste très restrictive et explicite de patterns qui correspondent uniquement à des titres
    # Ces patterns doivent être strictement limités aux formats de titres courants dans les contrats
    strict_title_patterns = [
        # Titres markdown (# Titre)
        r"^#+\s+\**[A-Z][A-Z\s]+\**$",
        # Formats explicites de sections numérotées
        r"^ARTICLE\s+[IVXLCDM]+\s*[:\.]",
        r"^SECTION\s+[0-9]+\s*[:\.]",
        r"^CHAPTER\s+[0-9]+\s*[:\.]",
        r"^ANNEXE?\s+[A-Z]:\s*",
        r"^APPENDIX\s+[A-Z]:\s*",
        # Titres numérotés tout en majuscules (très spécifique)
        r"^[0-9]+\.\s+[A-Z][A-Z\s]+$",
        # Nouveaux patterns plus précis pour les titres de contrats
        r"^#+\s+\**\d+\.\d+\s+[A-Z][A-Z\s]+\**$",  # ## 1.2 TITLE
        r"^#+\s+\**\d+\.\d+\.\d+\s+[A-Z][A-Z\s]+\**$",  # ## 1.2.3 TITLE
        r"^\*\*[A-Z][A-Z\s]+\*\*$",  # **TITLE**
        r"^[A-Z][A-Z\s]+:$",  # TITLE:
    ]

    # Fonction pour vérifier si un paragraphe est un titre
    def is_strict_title(paragraph):
        # Vérifier les patterns explicites de titre
        for pattern in strict_title_patterns:
            if re.match(pattern, paragraph):
                return True

        # Vérifier le cas spécial pour "# N. TITLE"
        if re.match(r"^#+\s+\**[0-9]+\.\s+[A-Z][A-Z\s]+\**$", paragraph):
            return True

        # Si le texte est court (<5 mots), tout en majuscules et pas de ponctuation finale,
        # c'est probablement un titre
        words = paragraph.split()
        if (
            len(words) <= 5
            and paragraph.isupper()
            and not paragraph.endswith((".", "!", "?", ",", ";", ":", ")", "]"))
        ):
            return True

        return False

    # Amélioration radicale: Utiliser une approche ultra permissive pour la détection du contenu supprimé
    def is_text_present_in_chunks(original_para, chunk_texts):
        """Vérifie si un texte est présent dans les chunks avec une comparaison extrêmement souple"""

        # Si le paragraphe est très court ou vide, considérer comme présent
        if not original_para.strip() or len(original_para.strip()) < 10:
            return True

        # Normalisation agressive des deux textes
        def normalize_text(text):
            # Supprimer tous les caractères spéciaux et la ponctuation
            text = re.sub(r"[^\w\s]", " ", text)
            # Supprimer les espaces multiples
            text = re.sub(r"\s+", " ", text)
            # Mettre en minuscules
            return text.lower().strip()

        # Texte original normalisé
        orig_norm = normalize_text(original_para)

        # Si le texte est très court (moins de 4 mots), considérer qu'il est probablement présent
        if len(orig_norm.split()) < 4:
            return True

        # Obtenir les mots significatifs du texte original (au moins 4 lettres)
        significant_words = [w for w in orig_norm.split() if len(w) >= 4]

        # Si pas de mots significatifs, considérer comme présent
        if not significant_words:
            return True

        # Vérifier l'intégralité du texte dans un seul chunk
        all_chunks_text = " ".join(chunk_texts)
        all_chunks_norm = normalize_text(all_chunks_text)

        # Si au moins 40% des mots significatifs sont présents dans l'ensemble des chunks,
        # considérer le texte comme présent (seuil très permissif)
        threshold = 0.4
        words_found = sum(
            1 for word in significant_words if word in all_chunks_norm.split()
        )
        if words_found / len(significant_words) >= threshold:
            return True

        # Pour les éléments de liste spécifiquement (très permissif)
        if original_para.strip().startswith(("-", "•", "*", "➢")) or re.match(
            r"^\d+[\.\)]", original_para.strip()
        ):
            # Extraire le contenu après le marqueur de liste
            list_content = re.sub(r"^[-•*➢\d\.\)]+\s*", "", original_para).strip()
            list_content_norm = normalize_text(list_content)

            # Si le contenu de la liste est court, le considérer comme présent
            if len(list_content_norm.split()) < 5:
                return True

            # Vérifier les mots significatifs du contenu de la liste
            list_words = [w for w in list_content_norm.split() if len(w) >= 4]
            if not list_words:
                return True

            # Si au moins 30% des mots significatifs sont présents, considérer comme présent (très permissif)
            words_found = sum(
                1 for word in list_words if word in all_chunks_norm.split()
            )
            if len(list_words) > 0 and words_found / len(list_words) >= 0.3:
                return True

            # Vérifier les premiers mots (peu importe leur longueur)
            first_few_words = list_content_norm.split()[:3]  # Premiers 3 mots
            if all(word in all_chunks_norm for word in first_few_words):
                return True

        # Dernière vérification: trouver des séquences de mots
        # Si on trouve une séquence de 2-3 mots consécutifs, considérer comme présent
        words = orig_norm.split()
        for i in range(len(words) - 1):
            if i < len(words) - 2:
                # Séquence de 3 mots
                seq = " ".join(words[i : i + 3])
                if seq in all_chunks_norm:
                    return True
            # Séquence de 2 mots
            seq = " ".join(words[i : i + 2])
            if seq in all_chunks_norm:
                return True

        # Vérification spéciale: tableaux et listes numérotées
        # Pour les lignes qui ressemblent à des entrées de tableaux ou des éléments de listes
        if (
            "|" in original_para
            or re.search(r"^\s*\d+\.\d+\s+", original_para)
            or re.search(r"^\s*[a-z]\)\s+", original_para)
        ):
            # Considérer très permissif, chercher juste quelques mots clés
            key_words = [w for w in orig_norm.split() if len(w) >= 5][
                :3
            ]  # Jusqu'à 3 mots longs
            if key_words and any(word in all_chunks_norm for word in key_words):
                return True

        return False

    # Trouver les paragraphes qui sont dans l'original mais pas dans les chunks
    removed_titles = []
    removed_content = []

    for i, para in enumerate(original_paragraphs):
        # Ignorer les paragraphes vides ou presque vides
        if not para.strip() or len(para.strip()) < 5:
            continue

        # Vérifier si le paragraphe est dans les chunks avec la fonction améliorée
        if not is_text_present_in_chunks(para, chunk_texts):
            # Ajouter le contexte (paragraphes avant et après)
            context = []
            if i > 0:
                context.append(f"Contexte précédent: {original_paragraphs[i-1]}")

            if i < len(original_paragraphs) - 1:
                context.append(f"Contexte suivant: {original_paragraphs[i+1]}")

            # Vérifier si c'est strictement un titre selon nos patterns très limités
            is_strict_title_result = is_strict_title(para)

            # Classifier uniquement les titres très évidents, laisser tout le reste comme "contenu"
            if is_strict_title_result:
                context.append(f"TITRE supprimé: {para}")
                removed_titles.append((context, "title"))
            else:
                context.append(f"CONTENU supprimé: {para}")
                removed_content.append((context, "content"))

    # Afficher les titres supprimés
    if removed_titles:
        print("\n📑 Titres supprimés:")
        print("-" * 40)
        for context, _ in removed_titles:
            for line in context:
                print(f"- {line}")
            print("-" * 40)

    # Afficher détail des contenus supprimés
    if removed_content:
        print(f"\n📄 {len(removed_content)} paragraphes supprimés")
        print("\n⚠️ Détail des paragraphes supprimés:")
        print("-" * 40)
        for context, _ in removed_content:
            for line in context:
                print(f"- {line}")
            print("-" * 40)

    # Statistiques
    print(f"\n📊 Statistiques du traitement:")
    print(f"- Nombre total de titres supprimés: {len(removed_titles)}")
    print(f"- Nombre total de paragraphes supprimés: {len(removed_content)}")

    if not removed_titles and not removed_content:
        print("Aucun contenu n'a été supprimé lors du découpage.")


def display_semantic_split_chunks(
    structure_chunks: List[Chunk], final_chunks: List[Chunk]
) -> None:
    """
    Affiche les chunks qui ont subi un découpage sémantique dans l'approche hybride

    Args:
        structure_chunks: Chunks initiaux après découpage structurel
        final_chunks: Chunks finaux après découpage sémantique
    """
    print("\n🔍 Chunks ayant subi un découpage sémantique:")
    print("=" * 80)

    # Identifier les chunks originaux qui ont été découpés
    split_chunks = []
    for original_chunk in structure_chunks:
        # Compter combien de chunks finaux proviennent de ce chunk original
        sub_chunks = [
            c for c in final_chunks if c.section_number == original_chunk.section_number
        ]
        if len(sub_chunks) > 1:  # Si le chunk a été découpé
            split_chunks.append((original_chunk, sub_chunks))

    if not split_chunks:
        print("Aucun chunk n'a subi de découpage sémantique.")
        return

    print(f"Nombre de chunks découpés sémantiquement: {len(split_chunks)}")

    for original_chunk, sub_chunks in split_chunks:
        print("\n" + "=" * 40)
        print(f"Chunk original:")
        print(f"Section: {original_chunk.section_number}")
        print(f"Hiérarchie: {' -> '.join(original_chunk.hierarchy)}")
        print(f"Taille originale: {len(original_chunk.content.split())} mots")
        print("\nDécoupé en {len(sub_chunks)} sous-chunks:")

        for i, sub_chunk in enumerate(sub_chunks, 1):
            print(f"\nSous-chunk {i}/{len(sub_chunks)}:")
            print(
                f"Position: {getattr(sub_chunk, 'position', 'N/A')}/{getattr(sub_chunk, 'total_chunks', 'N/A')}"
            )
            print(f"Taille: {len(sub_chunk.content.split())} mots")
            print(
                f"Contenu: {sub_chunk.content[:200]}..."
            )  # Afficher les 200 premiers caractères
        print("=" * 40)


def process_contract(filepath: str) -> List[Chunk]:
    """
    Process a contract file and return intelligent chunks using a hybrid approach:
    1. First split by legal structure (articles, sections, subsections)
    2. Then apply semantic chunking for sections exceeding 800 tokens
    3. Preserve hierarchical metadata for traceability

    Args:
        filepath: Path to the contract file
        use_semantic_chunking: Whether to use pure semantic chunking instead of the hybrid approach

    Returns:
        List of Chunk objects with preserved legal structure and metadata
    """
    print("\n🔄 Début du traitement du document...")
    start_time = time.time()

    # 1. Load and extract text from PDF
    print(
        "📄 Extraction du texte du PDF (avec détection des en-têtes/pieds de page et suppression des références d'images)..."
    )
    text, document_title = extract_text_contract(filepath)
    print(f"✅ Texte extrait ({len(text.split())} mots)")

    print("\n🔄 Découpage du texte avec approche hybride (structure + sémantique)...")
    # First split by legal structure
    splitter = IntelligentSplitter(document_title=document_title)
    structure_chunks = splitter.split(text)

    # Then apply semantic chunking for large sections
    semantic_manager = SemanticChunkManager(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.6,
        buffer_size=3,
        chunk_size=800,  # Limite de ~800 tokens
        chunk_overlap=100,  # Chevauchement de 100 tokens
    )

    chunks = []
    for chunk in structure_chunks:
        # If section is small enough, keep it as is
        if len(chunk.content.split()) < 800:  # Approximate token count
            chunks.append(chunk)
        else:
            # For large sections, apply semantic chunking
            sub_chunks = semantic_manager.chunk_text(chunk.content)
            # Preserve metadata in sub-chunks
            for sub_chunk in sub_chunks:
                sub_chunk.section_number = chunk.section_number
                sub_chunk.hierarchy = chunk.hierarchy
                sub_chunk.document_title = chunk.document_title
                sub_chunk.parent_section = chunk.parent_section
                sub_chunk.chapter_title = chunk.chapter_title
                # Add position metadata
                sub_chunk.position = len(chunks)
                sub_chunk.total_chunks = len(sub_chunks)
            chunks.extend(sub_chunks)

    # 3. Group chunks hierarchically
    print("\n🔍 Regroupement hiérarchique des chunks...")
    grouper = HierarchicalGrouper()
    hierarchical_groups = grouper.group_chunks(chunks)

    # 4. Initialize embeddings and ChromaDB
    print("\n🔍 Initialisation des embeddings et de ChromaDB...")
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

    # 5. Prepare chunks for ChromaDB with enhanced metadata
    print("\n📦 Préparation des chunks pour ChromaDB...")
    chroma_chunks = []
    for chunk in chunks:
        # Enhanced metadata structure
        metadata = {
            "section_number": chunk.section_number or "unknown",
            "hierarchy": chunk.hierarchy or ["unknown"],
            "document_title": chunk.document_title or "unknown",
            "parent_section": chunk.parent_section or "unknown",
            "chapter_title": chunk.chapter_title or "unknown",
            "title": document_title,
            "content": chunk.content,
            "chunk_type": (
                str(chunk.chunk_type) if hasattr(chunk, "chunk_type") else "unknown"
            ),
            "position": getattr(chunk, "position", None),
            "total_chunks": getattr(chunk, "total_chunks", None),
            "chunk_size": len(chunk.content.split()),  # Approximate token count
            "timestamp": time.time(),
        }

        # Enhanced content with metadata
        content = f"""
Section: {metadata['section_number']}
Hiérarchie complète: {' -> '.join(metadata['hierarchy'])}
Document: {metadata['document_title']}
Position: {metadata['position']}/{metadata['total_chunks']}

Contenu:
{chunk.content}
"""
        chroma_chunks.append({"content": content, "metadata": metadata})

    # 6. Add chunks to ChromaDB
    print("\n💾 Ajout des chunks à ChromaDB...")
    chroma_manager.add_documents(chroma_chunks)
    print("✅ Chunks ajoutés à ChromaDB")

    # Print document metadata
    print("\nDocument Metadata:")
    print(f"- Title: {document_title}")
    print(f"- Author: Unknown")
    print(f"- Pages: Unknown")

    # Print processing time and statistics
    processing_time = time.time() - start_time
    print(f"\n⏱️ Temps total de traitement: {processing_time:.2f} secondes")

    print(f"📊 Nombre de chunks créés: {len(chunks)}")
    print(
        f"📊 Taille moyenne des chunks: {sum(len(c.content.split()) for c in chunks) / len(chunks):.1f} tokens"
    )

    # Display chunks details and removed content
    display_chunks_details(chunks)
    display_removed_content(text, chunks)

    # Display semantic split chunks if in hybrid mode
    if structure_chunks:
        display_semantic_split_chunks(structure_chunks, chunks)

    return chunks


def search_contracts(query: str, n_results: int = 5) -> None:
    """
    Search in the contract database

    Args:
        query: Search query
        n_results: Number of results to return
    """
    print(f"\n🔍 Recherche: {query}")

    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

    # Search
    results = chroma_manager.search(query, n_results=n_results)

    # Display results
    print(f"\n📊 Résultats ({len(results)} trouvés):")
    for i, result in enumerate(results, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Section: {result['metadata']['section']}")
        print(f"Hiérarchie: {result['metadata']['hierarchy']}")
        print(f"Document: {result['metadata']['document_title']}")
        print(f"Contenu: {result['document'][:200]}...")
        print(f"Distance: {result['distance']:.4f}")


def chat_with_contract(query: str, n_context: int = 3) -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation

    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
    """
    print(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)

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
    from rag.ollama_chat import ask_ollama

    response = ask_ollama(prompt)
    print("\n🤖 Réponse :")
    print(response)

    # Display sources with metadata
    print("\n📚 Sources :")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print("\n" + "-" * 40)
        print(f"\nSource {i}/{len(results)}")
        print("-" * 40)

        print(f"Distance: {result['distance']:.4f}")

        # Afficher le contenu
        print(result["metadata"].get("content", result["document"])[:200] + "...")
        print("-" * 40)

    print(f"\n📊 Nombre total de sources: {len(results)}")


def hybrid_chunk_text(text, document_title):
    """Méthode hybride combinant chunking intelligent et sémantique"""
    # 1. Découper d'abord selon la structure (sections principales)
    splitter = IntelligentSplitter(document_title=document_title)
    structure_chunks = splitter.split(text)

    # 2. Pour chaque grande section, appliquer le chunking sémantique si nécessaire
    final_chunks = []
    for chunk in structure_chunks:
        # Si le chunk est petit, le garder tel quel
        if len(chunk.content) < 1000:
            final_chunks.append(chunk)
        else:
            # Pour les sections longues, appliquer chunking sémantique
            semantic_manager = SemanticChunkManager(
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.6,
                buffer_size=2,
            )
            # Préserver les métadonnées de la section dans les sous-chunks
            semantic_sub_chunks = semantic_manager.chunk_text(chunk.content)
            for sub_chunk in semantic_sub_chunks:
                sub_chunk.section_number = chunk.section_number
                sub_chunk.hierarchy = chunk.hierarchy
                sub_chunk.document_title = chunk.document_title
                sub_chunk.parent_section = chunk.parent_section
                sub_chunk.chapter_title = chunk.chapter_title
            final_chunks.extend(semantic_sub_chunks)

    return final_chunks


def calculate_optimal_threshold(text):
    """Calcule le seuil optimal basé sur la complexité du texte"""
    # Cette fonction est maintenant déléguée au SemanticChunkManager qui contient
    # une implémentation plus sophistiquée via _calculate_optimal_threshold

    from rag.semantic_chunker import SemanticChunkManager

    # Créer une instance temporaire pour utiliser la méthode
    temp_manager = SemanticChunkManager()

    # Utiliser la méthode améliorée du manager
    return temp_manager._calculate_optimal_threshold(text)


def preprocess_legal_text(text):
    """Prétraite le texte juridique pour préserver les clauses"""
    # Cette fonction est maintenant déléguée au SemanticChunkManager qui contient
    # une implémentation plus sophistiquée via _preprocess_text_with_section_markers
    # avec un ensemble élargi de patterns juridiques et de références croisées

    # Nous gardons ce code pour compatibilité, mais il utilise maintenant les patterns
    # du SemanticChunkManager pour une cohérence dans le traitement

    from rag.semantic_chunker import SemanticChunkManager

    # Créer une instance temporaire pour accéder aux patterns
    temp_manager = SemanticChunkManager()

    # Traiter le texte en utilisant les patterns du manager
    processed_lines = []
    for line in text.split("\n"):
        if any(re.search(pattern, line) for pattern in temp_manager.legal_patterns):
            processed_lines.append("[CLAUSE_START]" + line)
        elif any(
            re.search(pattern, line) for pattern in temp_manager.cross_ref_patterns
        ):
            processed_lines.append("[CROSS_REF]" + line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print(
            "Usage: python main.py <contract_file> [--semantic-chunking] [search_query|--chat]"
        )
        sys.exit(1)

    filepath = sys.argv[1]

    # If --chat is provided, enter chat mode
    if len(sys.argv) > 2 and sys.argv[2] == "--chat":
        print("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nVotre question : ")
            if query.lower() == "exit":
                break
            chat_with_contract(query)
    # If search query is provided, perform search
    elif len(sys.argv) > 2:
        search_query = " ".join(sys.argv[2:])
        search_contracts(search_query)
    else:
        # Process the contract
        chunks = process_contract(filepath)
