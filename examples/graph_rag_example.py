#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphRAG Example - Demonstrates how to use the Graph-based Retrieval Augmented Generation

This example script shows how to:
1. Process a contract document with GraphRAG
2. Visualize the document graph
3. Perform searches with GraphRAG
4. Chat with the document using GraphRAG
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import networkx as nx

# Configure paths properly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import after path configuration
from rag.embeddings_manager import EmbeddingsManager
from rag.graph_builder import GraphBuilder
from rag.graph_rag import GraphRAG
from rag.chroma_manager import ChromaDBManager
from main import process_contract


def visualize_document_graph(graph_rag):
    """
    Visualize the document graph.
    
    Args:
        graph_rag: GraphRAG instance with populated graph
    """
    print("\n📊 Visualisation du graphe du document...")
    
    if not graph_rag.graph or len(graph_rag.graph.nodes) == 0:
        print("❌ Le graphe est vide. Veuillez d'abord traiter un document.")
        return
    
    # Create a simplified graph for visualization (limit to 30 nodes max)
    G = graph_rag.graph
    if len(G.nodes) > 30:
        print(f"⚠️ Le graphe est trop grand ({len(G.nodes)} nœuds). Affichage des 30 premiers nœuds.")
        nodes = list(G.nodes)[:30]
        G = G.subgraph(nodes)
    
    # Print graph details for debugging
    print(f"📊 Détails du graphe:")
    print(f"Nombre de nœuds: {len(G.nodes)}")
    print(f"Nombre d'arêtes: {len(G.edges)}")
    print(f"Nœuds: {list(G.nodes)[:10]}{'...' if len(G.nodes) > 10 else ''}")
    
    plt.figure(figsize=(12, 8))
    
    # Use different colors for different types of edges
    edge_colors = []
    edge_labels = {}
    
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'unknown')
        if relation == 'next':
            edge_colors.append('blue')
        elif relation == 'previous':
            edge_colors.append('green')
        elif relation == 'references':
            edge_colors.append('red')
            edge_labels[(u, v)] = 'ref'
        else:
            edge_colors.append('gray')
    
    # Position nodes using a hierarchical layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color=edge_colors, 
                          arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # Draw edge labels for references
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Structure hiérarchique du document avec références croisées")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("document_graph.png", dpi=300, bbox_inches='tight')
    print("✅ Graphe visualisé et sauvegardé dans 'document_graph.png'")
    plt.show()


def force_process_contract(filepath):
    """
    Force processing of the contract even if it's already in ChromaDB.
    This helps ensure we have proper metadata for graph building.
    
    Args:
        filepath: Path to the contract file
        
    Returns:
        The processed chunks
    """
    print(f"🔄 Traitement forcé du document {filepath}...")
    
    # First clear the existing ChromaDB collection
    embeddings_manager = EmbeddingsManager()
    chroma_manager = ChromaDBManager(embeddings_manager)
    try:
        chroma_manager.collection.delete(where={})
        print("✅ Collection ChromaDB nettoyée")
    except Exception as e:
        print(f"⚠️ Erreur lors du nettoyage de ChromaDB: {e}")
    
    # Now process the document
    chunks = process_contract(filepath)
    print(f"✅ Document traité avec {len(chunks)} chunks")
    
    # Check if processing was successful
    if not chunks or len(chunks) == 0:
        print("❌ Erreur: Aucun chunk n'a été créé lors du traitement du document.")
        sys.exit(1)
    
    # Verify metadata for graph building
    print("\n🔍 Vérification des métadonnées pour la construction du graphe...")
    has_section_numbers = sum(1 for c in chunks if c.section_number) / len(chunks) if chunks else 0
    has_hierarchy = sum(1 for c in chunks if c.hierarchy) / len(chunks) if chunks else 0
    
    print(f"- Chunks avec numéros de section: {has_section_numbers:.1%}")
    print(f"- Chunks avec hiérarchie: {has_hierarchy:.1%}")
    
    if has_section_numbers < 0.5 or has_hierarchy < 0.5:
        print("⚠️ Attention: Moins de 50% des chunks ont des métadonnées de structure.")
        print("   Le graphe pourrait être incomplet.")
    
    return chunks


def debug_graph_construction(chroma_manager):
    """
    Debug the graph construction by examining the metadata in ChromaDB.
    
    Args:
        chroma_manager: ChromaDBManager instance
    """
    print("\n🔍 Débogage de la construction du graphe...")
    
    # Get all documents from ChromaDB
    results = chroma_manager.collection.get(include=["metadatas"])
    
    if not results['metadatas']:
        print("❌ Aucun document trouvé dans ChromaDB.")
        return
    
    # Check key metadata fields needed for graph construction
    section_numbers = [m.get('section_number') for m in results['metadatas']]
    hierarchies = [m.get('hierarchy') for m in results['metadatas']]
    
    print(f"Nombre total de documents: {len(results['metadatas'])}")
    print(f"Documents avec numéro de section: {sum(1 for s in section_numbers if s)}")
    print(f"Documents avec hiérarchie: {sum(1 for h in hierarchies if h)}")
    
    # Print a sample of metadata
    print("\nÉchantillon de métadonnées:")
    for i, metadata in enumerate(results['metadatas'][:3]):
        print(f"\nDocument {i+1}:")
        print(f"  Section: {metadata.get('section_number', 'Non spécifié')}")
        print(f"  Hiérarchie: {metadata.get('hierarchy', 'Non spécifié')}")
        print(f"  Parent: {metadata.get('parent_section', 'Non spécifié')}")


def main():
    """Main function to demonstrate GraphRAG functionality"""
    if len(sys.argv) < 2:
        print("Usage: python graph_rag_example.py <contract_file> [--search <query> | --chat | --force-process]")
        sys.exit(1)
    
    contract_file = sys.argv[1]
    force_processing = "--force-process" in sys.argv
    
    # Initialize embedding manager
    embeddings_manager = EmbeddingsManager()
    
    # Process the contract if needed or forced
    chroma_manager = ChromaDBManager(embeddings_manager)
    
    if force_processing or chroma_manager.collection.count() == 0:
        chunks = force_process_contract(contract_file)
    else:
        print(f"✅ Document déjà traité et stocké dans ChromaDB avec {chroma_manager.collection.count()} chunks.")
        chunks = None
    
    # Debug the metadata in ChromaDB
    debug_graph_construction(chroma_manager)
    
    # Initialize GraphRAG
    print("\n🔄 Initialisation de GraphRAG...")
    start_time = time.time()
    graph_rag = GraphRAG(embeddings_manager=embeddings_manager, chroma_manager=chroma_manager)
    print(f"✅ GraphRAG initialisé en {time.time() - start_time:.2f} secondes")
    
    # Visualize the graph
    visualize_document_graph(graph_rag)
    
    # Perform search or chat based on command line arguments
    if "--search" in sys.argv:
        query_index = sys.argv.index("--search") + 1
        if query_index < len(sys.argv):
            query = " ".join(sys.argv[query_index:])
            print(f"\n🔍 Recherche avec GraphRAG: {query}")
            
            # Compare standard search vs GraphRAG
            print("\n📊 Comparaison des résultats:")
            print("\n--- Recherche standard ---")
            from main import search_contracts
            search_contracts(query, n_results=3, use_graph=False)
            
            print("\n--- Recherche avec GraphRAG ---")
            search_contracts(query, n_results=3, use_graph=True, max_hop=2)
    
    elif "--chat" in sys.argv:
        print("\n💬 Mode chat avec GraphRAG activé. Tapez 'exit' ou 'quit' pour quitter.")
        while True:
            query = input("\nVotre question: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            from main import chat_with_contract
            chat_with_contract(query, n_context=3, use_graph=True, max_hop=2)
    

if __name__ == "__main__":
    main() 