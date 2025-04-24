import re
import sys
import time
from typing import List
from pathlib import Path

from rag.chroma_manager import ChromaDBManager
from rag.embeddings_manager import EmbeddingsManager
from rag.hierarchical_grouper import HierarchicalGrouper
from rag.intelligent_splitter import Chunk, IntelligentSplitter
from rag.pdf_loader import extract_text_contract
from rag.semantic_chunker import SemanticChunkManager
from rag.graph_manager import GraphManager


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
    Affiche TOUTES les lignes du texte original qui ne sont pas présentes dans les chunks.
    Utilise une comparaison ligne par ligne stricte.
    
    Args:
        original_text: Texte original du document
        chunks: Liste des chunks après découpage
    """
    print("\n🗑️ Contenu supprimé lors du découpage:")
    print("=" * 80)
    
    # Extraire toutes les lignes du texte original, avec normalisation minimale
    original_lines = []
    for line in original_text.split("\n"):
        line = line.strip()
        if line:  # Ignorer les lignes vides
            original_lines.append(line)
    
    # Extraire toutes les lignes des chunks
    chunk_lines = []
    for chunk in chunks:
        for line in chunk.content.split("\n"):
            line = line.strip()
            if line:  # Ignorer les lignes vides
                chunk_lines.append(line)
    
    # Liste très restrictive de patterns qui correspondent uniquement à des titres
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
    
    # Fonction pour vérifier si une ligne est un titre
    def is_strict_title(line):
        # Vérifier les patterns explicites de titre
        for pattern in strict_title_patterns:
            if re.match(pattern, line):
                return True
                
        # Vérifier le cas spécial pour "# N. TITLE"
        if re.match(r"^#+\s+\**[0-9]+\.\s+[A-Z][A-Z\s]+\**$", line):
            return True
            
        # Si le texte est court (<5 mots), tout en majuscules et pas de ponctuation finale,
        # c'est probablement un titre
        words = line.split()
        if (
            len(words) <= 5
            and line.isupper()
            and not line.endswith((".", "!", "?", ",", ";", ":", ")", "]"))
        ):
            return True
            
        return False
    
    # Comparer chaque ligne originale avec les lignes des chunks
    # Une ligne est considérée présente si elle est exactement dans les chunks (ou très légèrement différente)
    def is_line_present(line, chunk_lines):
        # Normalisation minimale
        def normalize_for_comparison(text):
            # Supprime juste les espaces en début/fin et réduit les espaces multiples
            return re.sub(r"\s+", " ", text).strip()
        
        normalized_line = normalize_for_comparison(line)
        
        # Vérification exacte
        for chunk_line in chunk_lines:
            normalized_chunk_line = normalize_for_comparison(chunk_line)
            if normalized_line == normalized_chunk_line:
                return True
            
            # Vérification avec légère tolérance pour les espaces/tirets/points
            # Remplacer les caractères spéciaux par des espaces et comparer
            clean_line = re.sub(r"[-_.,;:()]", " ", normalized_line)
            clean_line = re.sub(r"\s+", " ", clean_line).strip()
            
            clean_chunk = re.sub(r"[-_.,;:()]", " ", normalized_chunk_line)
            clean_chunk = re.sub(r"\s+", " ", clean_chunk).strip()
            
            if clean_line == clean_chunk:
                return True
                
        return False
    
    # Trouver les lignes qui ne sont pas dans les chunks
    missing_titles = []
    missing_content = []
    
    for i, line in enumerate(original_lines):
        if not is_line_present(line, chunk_lines):
            # Collecter le contexte (ligne précédente et suivante)
            context = []
            if i > 0:
                context.append(f"Ligne précédente: {original_lines[i-1]}")
            
            if i < len(original_lines) - 1:
                context.append(f"Ligne suivante: {original_lines[i+1]}")
            
            # Vérifier si c'est un titre
            if is_strict_title(line):
                context.append(f"TITRE supprimé: {line}")
                missing_titles.append((context, line))
            else:
                context.append(f"LIGNE supprimée: {line}")
                missing_content.append((context, line))
    
    # Afficher les titres supprimés
    if missing_titles:
        print("\n📑 Titres supprimés:")
        print("-" * 40)
        for context, title in missing_titles:
            for line in context:
                print(f"- {line}")
            print("-" * 40)
    
    # Afficher les lignes de contenu supprimées
    if missing_content:
        print(f"\n📄 {len(missing_content)} lignes supprimées")
        print("\n⚠️ Détail des lignes supprimées:")
        print("-" * 40)
        for context, content in missing_content:
            for line in context:
                print(f"- {line}")
            print("-" * 40)
    
    # Statistiques
    print(f"\n📊 Statistiques du traitement:")
    print(f"- Nombre total de titres supprimés: {len(missing_titles)}")
    print(f"- Nombre total de lignes supprimées: {len(missing_content)}")
    
    if not missing_titles and not missing_content:
        print("Aucune ligne n'a été supprimée lors du découpage.")


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
    4. Apply post-processing to restore important legal content that might have been lost

    Args:
        filepath: Path to the contract file

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

    # 2.5 Post-traitement: restaurer le contenu juridique important qui aurait pu être perdu
    print("\n🔄 Application du post-traitement pour restaurer le contenu juridique important...")
    chunks = restore_important_content(text, chunks)

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
    
    # 7. Build knowledge graph from chunks
    print("\n🔄 Construction du graphe de connaissances...")
    graph_manager = GraphManager(chroma_manager, embeddings_manager)
    graph = graph_manager.build_graph(chroma_chunks)
    print(f"✅ Graphe de connaissances créé avec {len(graph.nodes)} nœuds et {len(graph.edges)} relations")
    
    # Generate and save graph visualization
    print("\n🎨 Génération de la visualisation du graphe...")
    graph_output_path = f"graph_{Path(filepath).stem}.png"
    graph_manager.visualize_graph(graph, output_path=graph_output_path)
    
    # Export detailed graph information to text file
    print("\n📝 Export des détails du graphe de connaissances...")
    details_output_path = f"graph_details_{Path(filepath).stem}.txt"
    graph_manager.export_graph_details(graph, output_path=details_output_path)

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


def chat_with_contract(query: str, n_context: int = 3, use_graph: bool = True) -> None:
    """
    Chat with the contract using embeddings for context and Ollama for generation
    
    Args:
        query: User's question
        n_context: Number of relevant chunks to use as context
        use_graph: Whether to use graph-based context expansion
    """
    print(f"\n💬 Chat: {query}")

    # Initialize managers
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)
    
    # Initialize graph manager if using graph augmentation
    graph_manager = None
    knowledge_graph = None
    if use_graph:
        print("🔍 Utilisation du graphe de connaissances pour enrichir le contexte...")
        graph_manager = GraphManager(chroma_manager, embeddings_manager)
        # Load or build the graph
        knowledge_graph = load_or_build_graph(chroma_manager, embeddings_manager)
    
    # Search for relevant context
    results = chroma_manager.search(query, n_results=n_context)
    
    # Augment results with graph traversal if enabled
    if use_graph and knowledge_graph:
        graph_results = get_graph_augmented_results(knowledge_graph, results, n_additional=2)
        # Combine results (ensuring no duplicates)
        combined_results = merge_results(results, graph_results)
    else:
        combined_results = results
    
    # Prepare context for the prompt
    context = "\n\n".join(
        [
            f"Document: {result['metadata'].get('document_title', 'Non spécifié')}\n"
            f"Section: {result['metadata'].get('section_number', 'Non spécifié')}\n"
            f"Chapter: {result['metadata'].get('chapter_title', 'Non spécifié')}\n"
            f"Content: {result['document']}"
            for result in combined_results
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
    for i, result in enumerate(combined_results, 1):
        print("\n" + "-" * 40)
        print(f"\nSource {i}/{len(combined_results)}")
        
        # Indicate if this source came from graph traversal
        if result.get("source_type") == "graph":
            print("📊 Source obtenue via le graphe de connaissances")
            print(f"Relation: {result.get('relation_type', 'Non spécifié')}")
        
        print("-" * 40)
        print(f"Distance: {result.get('distance', 'N/A'):.4f}")

        # Afficher le contenu
        print(result["metadata"].get("content", result["document"])[:200] + "...")
        print("-" * 40)

    print(f"\n📊 Nombre total de sources: {len(combined_results)}")


def load_or_build_graph(chroma_manager, embeddings_manager):
    """
    Load existing graph or build a new one if needed
    
    Args:
        chroma_manager: ChromaDB manager instance
        embeddings_manager: Embeddings manager instance
    
    Returns:
        KnowledgeGraph: The loaded or built knowledge graph
    """
    import os
    import pickle
    
    graph_path = "knowledge_graph.pkl"
    
    # Check if graph file exists
    if os.path.exists(graph_path):
        try:
            print(f"📂 Chargement du graphe existant depuis {graph_path}...")
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Erreur lors du chargement du graphe: {str(e)}")
    
    # If no graph exists or loading failed, build a new one
    print("🔄 Construction d'un nouveau graphe de connaissances...")
    
    # Get all documents from ChromaDB
    all_docs = chroma_manager.get_all_documents()
    
    # Build graph
    graph_manager = GraphManager(chroma_manager, embeddings_manager)
    graph = graph_manager.build_graph(all_docs)
    
    # Save graph for future use
    try:
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"✅ Graphe sauvegardé dans {graph_path}")
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder le graphe: {str(e)}")
    
    return graph


def get_graph_augmented_results(graph, initial_results, n_additional=2):
    """
    Expand search results by traversing the knowledge graph
    
    Args:
        graph: Knowledge graph
        initial_results: Initial results from vector search
        n_additional: Number of additional results to retrieve
        
    Returns:
        List of additional results from graph traversal
    """
    graph_results = []
    processed_ids = set()
    
    # Add initial result IDs to processed set to avoid duplicates
    for result in initial_results:
        if 'id' in result:
            processed_ids.add(result['id'])
    
    # Find nodes that correspond to initial results
    for result in initial_results:
        # Try to find matching node by content or metadata
        result_content = result.get('document', '')
        
        # Find node that most closely matches this result
        matching_node_id = None
        for node_id, node in graph.nodes.items():
            if result_content in node.content:
                matching_node_id = node_id
                break
        
        if not matching_node_id:
            continue
            
        # Find related nodes through different relation types
        for edge in graph.edges:
            if edge.source == matching_node_id:
                target_node_id = edge.target
                
                # Skip if already processed
                if target_node_id in processed_ids:
                    continue
                    
                processed_ids.add(target_node_id)
                
                # Get the target node and create a result
                target_node = graph.nodes[target_node_id]
                
                # Create a result object similar to the ChromaDB results
                graph_result = {
                    'document': target_node.content,
                    'metadata': target_node.metadata,
                    'source_type': 'graph',
                    'relation_type': edge.relation_type,
                    'distance': 1.0 - edge.weight,  # Convert weight to distance
                    'id': target_node_id
                }
                
                graph_results.append(graph_result)
                
                # Limit number of additional results
                if len(graph_results) >= n_additional:
                    return graph_results
    
    return graph_results


def merge_results(vector_results, graph_results):
    """
    Merge vector-based and graph-based results, avoiding duplicates
    
    Args:
        vector_results: Results from vector search
        graph_results: Results from graph traversal
        
    Returns:
        List of combined results
    """
    # Start with vector results
    combined_results = list(vector_results)
    
    # Track document contents to avoid duplicates
    seen_contents = set(result.get('document', '')[:100] for result in vector_results)
    
    # Add graph results that aren't duplicates
    for result in graph_results:
        content_preview = result.get('document', '')[:100]
        if content_preview not in seen_contents:
            seen_contents.add(content_preview)
            combined_results.append(result)
    
    return combined_results


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


def restore_important_content(original_text: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    Fonction de post-traitement qui identifie les lignes juridiques importantes
    qui ont été supprimées et les réintègre dans les chunks appropriés.
    NE restaure PAS les titres, mais restaure toutes les lignes de contenu juridique.
    
    Args:
        original_text: Texte original du document
        chunks: Liste des chunks après découpage initial
        
    Returns:
        Liste des chunks avec le contenu important restauré
    """
    print("\n🔄 Post-traitement: recherche de contenu juridique important supprimé...")
    
    # Extraire toutes les lignes du texte original
    original_lines = []
    for line in original_text.split("\n"):
        line = line.strip()
        if line:  # Ignorer les lignes vides
            original_lines.append(line)
    
    # Extraire toutes les lignes des chunks
    chunk_lines = []
    for chunk in chunks:
        for line in chunk.content.split("\n"):
            line = line.strip()
            if line:  # Ignorer les lignes vides
                chunk_lines.append(line)
    
    # Fonction pour normaliser les lignes avant comparaison
    def normalize_line(text):
        return re.sub(r"\s+", " ", text).strip()
    
    # Identifier les lignes manquantes
    missing_lines = []
    for line in original_lines:
        normalized_line = normalize_line(line)
        found = False
        
        for chunk_line in chunk_lines:
            normalized_chunk = normalize_line(chunk_line)
            if normalized_line == normalized_chunk:
                found = True
                break
                
        if not found:
            missing_lines.append(line)
    
    # Fonction pour détecter si une ligne est un titre
    def is_title(line):
        # Patterns pour identifier les titres
        title_patterns = [
            # Titres markdown (# Titre)
            r"^#+\s+\**[A-Za-z0-9]",
            # Formats explicites de sections numérotées
            r"^ARTICLE\s+[IVXLCDM]+\s*[:\.]",
            r"^SECTION\s+[0-9]+\s*[:\.]",
            r"^CHAPTER\s+[0-9]+\s*[:\.]",
            r"^ANNEXE?\s+[A-Z]:",
            r"^APPENDIX\s+[A-Z]:",
            # Titres numérotés
            r"^[0-9]+(\.[0-9]+)*\s+[A-Z]",
            # Titres avec des caractères spéciaux
            r"^\*\*[A-Z]",
            r"^[A-Z][A-Z\s]+:$",
            # Titres courts tout en majuscules
            r"^[A-Z][A-Z\s]{1,30}$"
        ]
        
        # Vérifier si la ligne correspond à un des patterns de titre
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True
                
        # Autres indices de titres
        if line.isupper() and len(line.split()) <= 5:
            return True
            
        return False
    
    # Critères pour identifier les lignes juridiques importantes - APPROCHE TRÈS PERMISSIVE
    def is_important_legal_content(line):
        # D'abord vérifier si c'est un titre - si oui, ce n'est pas du contenu juridique à restaurer
        if is_title(line):
            return False
            
        # Mots-clés juridiques importants (LISTE ÉTENDUE)
        legal_keywords = [
            # Termes juridiques standards
            "notwithstanding", "shall be", "exclusive remedy", 
            "sole and exclusive", "right to terminate", "limitation of liability",
            "indemnify", "warranty", "warranties", "liabilities", "liability",
            "remedies", "remedy", "disclaims", "disclaim", "claims", "claim",
            "damages", "damage", "breach", "termination", "terminate",
            "force majeure", "intellectual property", "confidential",
            "liquidated damages", "penalties", "penalty", 
            
            # Termes contractuels additionnels
            "shall", "obligation", "obligations", "responsibility", "responsibilities",
            "rights", "right", "terms", "conditions", "provisions", "stipulations",
            "agreement", "contract", "hereof", "herein", "thereof", "therein",
            "delivery", "deliver", "payment", "pay", "price", "fee", "fees",
            "delay", "delays", "timely", "schedule", "schedules", "deadline", 
            "deadline", "milestones", "milestone", "completion", "complete",
            "acceptance", "accepts", "accept", "approved", "approve", "approval",
            "rejected", "reject", "rejection", "dispute", "disputes", "resolution",
            "test", "testing", "inspection", "inspect", "audit", "review",
            "pursuant", "accordance", "compliance", "comply", "applicable",
            "indemnification", "indemnify", "indemnified", "indemnities",
            "insurance", "insured", "coverage", "purchaser", "supplier", "parties"
        ]
        
        # Expressions régulières pour clauses spécifiques
        clause_references = [
            r"clause\s+\d+(\.\d+)?", r"article\s+\d+(\.\d+)?",
            r"section\s+\d+(\.\d+)?", r"pursuant to", r"in accordance with",
            r"subject to", r"appendix [a-z]", r"annex [a-z]"
        ]
        
        # Vérifier les mots-clés
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in legal_keywords):
            return True
            
        # Vérifier les références à des clauses
        if any(re.search(pattern, line_lower) for pattern in clause_references):
            return True
            
        # Reconnaître les clauses de limitation ou d'exclusion
        exclusion_patterns = [
            r"not be liable", r"no liability", r"shall not",
            r"exclude[sd]?", r"except", r"exempted", r"limitation", 
            r"limited to", r"restrict(ed|ion)", r"waive[sd]?", r"waiver"
        ]
        
        if any(re.search(pattern, line_lower) for pattern in exclusion_patterns):
            return True
            
        # Structure conditionnelle typique des clauses contractuelles
        if re.search(r"if\s+.*\s+(shall|may|must|will|is|are)\s+", line_lower):
            return True
            
        # Si la ligne contient des termes d'obligations, de conditions ou de conséquences
        if re.search(r"(shall|may|must|will)\s+.*\s+(if|unless|until|provided that)", line_lower):
            return True
            
        # Lignes qui commencent par des termes d'obligation contractuelle
        if re.search(r"^(the\s+)?(purchaser|supplier|contractor|client|party|parties)\s+(shall|may|will|must)", line_lower):
            return True
            
        # Lignes qui parlent de documents, livraisons, ou paiements
        if re.search(r"(documents?|delivery|payment|invoice|fee|compensation|reimbursement)", line_lower):
            return True
            
        return False
    
    # Identifier les lignes juridiques importantes parmi les lignes manquantes
    important_lines = [line for line in missing_lines if is_important_legal_content(line)]
    
    if not important_lines:
        print("✅ Aucun contenu juridique important n'a été supprimé.")
        return chunks
        
    print(f"🔍 {len(important_lines)} lignes de contenu juridique important identifiées pour restauration.")
    
    # Fonction pour trouver le meilleur chunk pour restaurer une ligne
    def find_best_chunk(line, chunks):
        # Trouver le contexte de la ligne dans le texte original
        line_index = original_lines.index(line)
        context_before = original_lines[max(0, line_index-5):line_index]
        context_after = original_lines[line_index+1:min(len(original_lines), line_index+6)]
        
        best_chunk = None
        best_score = -1
        
        for chunk in chunks:
            score = 0
            chunk_content = chunk.content.lower()
            
            # Vérifier si des lignes du contexte sont dans ce chunk
            for ctx_line in context_before + context_after:
                if normalize_line(ctx_line.lower()) in normalize_line(chunk_content):
                    score += 3  # Augmenter le poids du contexte
            
            # Vérifier si le chunk contient des mots-clés de la même section
            line_words = set(line.lower().split())
            chunk_words = set(chunk_content.split())
            common_words = line_words.intersection(chunk_words)
            score += len(common_words) * 0.2
            
            # Vérifier si le numéro de section correspond
            if hasattr(chunk, 'section_number') and chunk.section_number:
                # Extraire des numéros potentiels de section depuis la ligne
                section_matches = re.findall(r"clause\s+(\d+(\.\d+)?)", line.lower())
                if section_matches:
                    for match in section_matches:
                        if match[0] in chunk.section_number:
                            score += 3
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        # Si aucun chunk n'a un bon score, prendre celui qui a la meilleure correspondance textuelle
        if best_score <= 1:
            highest_score = -1
            best_matching_chunk = None
            
            for chunk in chunks:
                chunk_content = chunk.content.lower()
                
                # Si la ligne fait partie d'une section numérotée, essayer de trouver cette section
                section_match = re.search(r"\b(\d+(\.\d+)?)\b", line.lower())
                if section_match and section_match.group(1) in chunk_content:
                    return chunk
                
                # Sinon, utiliser la correspondance textuelle
                line_words = set(line.lower().split())
                chunk_words = set(chunk_content.split())
                common_words = line_words.intersection(chunk_words)
                
                if len(line_words) > 0:
                    score = len(common_words) / len(line_words)
                    
                    if score > highest_score:
                        highest_score = score
                        best_matching_chunk = chunk
            
            if best_matching_chunk:
                return best_matching_chunk
        
        return best_chunk
    
    # Restaurer les lignes importantes dans les chunks appropriés
    restored_chunks = list(chunks)  # Copie pour éviter de modifier l'original
    for line in important_lines:
        best_chunk = find_best_chunk(line, restored_chunks)
        if best_chunk:
            # Ajouter la ligne au chunk (à la fin)
            best_chunk.content = best_chunk.content + "\n\n" + line
            print(f"✅ Ligne restaurée dans un chunk approprié: {line[:60]}...")
        else:
            # Si aucun chunk approprié n'est trouvé, créer un nouveau chunk
            print(f"⚠️ Création d'un nouveau chunk pour la ligne: {line[:60]}...")
            new_chunk = Chunk(
                content=line, 
                section_number="unknown",
                hierarchy=["restored_content"],
                document_title=chunks[0].document_title if chunks else "unknown",
                parent_section="Restored Content",
                chapter_title="Restored Legal Content"
            )
            restored_chunks.append(new_chunk)
    
    print("✅ Post-traitement terminé. Contenu juridique important restauré.")
    return restored_chunks


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print(
            "Usage: python main.py <contract_file> [search_query|--chat|--graph-chat]"
            "\n       python main.py --search <query>"
            "\n       python main.py --chat"
            "\n       python main.py --graph-chat"
        )
        sys.exit(1)

    # If first argument is a command, handle it directly
    if sys.argv[1].startswith('--'):
        command = sys.argv[1]
        
        # Search mode
        if command == "--search" and len(sys.argv) > 2:
            search_query = " ".join(sys.argv[2:])
            search_contracts(search_query)
            sys.exit(0)
        
        # Chat modes without processing a document first
        elif command == "--chat":
            print("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
            while True:
                query = input("\nVotre question : ")
                if query.lower() == "exit":
                    break
                chat_with_contract(query, use_graph=False)
            sys.exit(0)
            
        elif command == "--graph-chat":
            print("\n🔍 Mode chat augmenté par graphe de connaissances activé. Tapez 'exit' pour quitter.")
            while True:
                query = input("\nVotre question : ")
                if query.lower() == "exit":
                    break
                chat_with_contract(query, use_graph=True)
            sys.exit(0)

    # Existing behavior for processing a contract
    filepath = sys.argv[1]

    # If --chat is provided, enter chat mode
    if len(sys.argv) > 2 and sys.argv[2] == "--chat":
        print("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nVotre question : ")
            if query.lower() == "exit":
                break
            chat_with_contract(query, use_graph=False)
    # If --graph-chat is provided, enter graph-augmented chat mode
    elif len(sys.argv) > 2 and sys.argv[2] == "--graph-chat":
        print("\n🔍 Mode chat augmenté par graphe de connaissances activé. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nVotre question : ")
            if query.lower() == "exit":
                break
            chat_with_contract(query, use_graph=True)
    # If search query is provided, perform search
    elif len(sys.argv) > 2:
        search_query = " ".join(sys.argv[2:])
        search_contracts(search_query)
    else:
        # Process the contract
        chunks = process_contract(filepath)