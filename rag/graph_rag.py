import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from rag.graph_builder import GraphBuilder
from rag.chroma_manager import ChromaDBManager
from rag.embeddings_manager import EmbeddingsManager


class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation system.
    
    This class integrates vector similarity search with graph traversal
    to provide more contextually relevant document retrieval.
    """
    
    def __init__(
        self,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        chroma_manager: Optional[ChromaDBManager] = None,
    ):
        """
        Initialize the GraphRAG system.
        
        Args:
            embeddings_manager: Optional custom embeddings manager
            chroma_manager: Optional custom ChromaDB manager
        """
        # Initialize components
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self.chroma_manager = chroma_manager or ChromaDBManager(self.embeddings_manager)
        self.graph_builder = GraphBuilder()
        self.graph = None
        self.document_to_node = {}
        self.node_to_document = {}
        
        # Build graph from existing chunks in ChromaDB
        self._build_graph_from_chroma()
    
    def _build_graph_from_chroma(self) -> None:
        """
        Build a graph structure from chunks already stored in ChromaDB.
        Creates connections based on hierarchical relationships.
        """
        # Get all documents from ChromaDB
        results = self.chroma_manager.collection.get(include=["metadatas", "documents"])
        
        if not results['metadatas']:
            print("No documents found in ChromaDB. Graph will be empty.")
            self.graph = nx.DiGraph()
            return
            
        # Build the graph
        self.graph = self.graph_builder.build_graph(results['metadatas'])
        
        # Create mappings between documents and graph nodes
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            node_id = metadata.get('section_number', f"unknown_{i}")
            self.document_to_node[doc] = node_id
            self.node_to_document[node_id] = doc
        
        # Add edges for sequential relationships (previous/next chunk)
        self._add_sequential_edges(results['metadatas'])
        
        # Add cross-reference edges based on content analysis
        self._add_crossreference_edges(results['documents'], results['metadatas'])
        
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def _add_sequential_edges(self, metadatas: List[Dict]) -> None:
        """
        Add edges for sequential relationships between chunks.
        
        Args:
            metadatas: List of chunk metadata dictionaries
        """
        # Group by parent section
        sections_by_parent = defaultdict(list)
        for metadata in metadatas:
            parent = metadata.get('parent_section', 'unknown')
            section = metadata.get('section_number', 'unknown')
            if parent != 'unknown' and section != 'unknown':
                sections_by_parent[parent].append((section, metadata))
        
        # Sort sections within each parent and add sequential edges
        for parent, sections in sections_by_parent.items():
            # Sort by section number if possible
            try:
                sorted_sections = sorted(sections, key=lambda x: float(x[0].split('.')[-1]) 
                                        if '.' in x[0] else float(x[0]))
            except (ValueError, TypeError):
                sorted_sections = sections  # Keep original order if sorting fails
            
            # Add sequential edges
            for i in range(len(sorted_sections) - 1):
                current_section = sorted_sections[i][0]
                next_section = sorted_sections[i + 1][0]
                if current_section in self.graph.nodes and next_section in self.graph.nodes:
                    self.graph.add_edge(current_section, next_section, relation="next")
                    self.graph.add_edge(next_section, current_section, relation="previous")
    
    def _add_crossreference_edges(self, documents: List[str], metadatas: List[Dict]) -> None:
        """
        Add edges for cross-references between sections.
        
        Args:
            documents: List of document contents
            metadatas: List of document metadata dictionaries
        """
        # Common patterns for cross-references in legal documents
        import re
        ref_patterns = [
            r"(?:see|refer to|as (?:defined|specified|described) in)\s+(?:section|article|clause)\s+(\d+(?:\.\d+)*)",
            r"(?:section|article|clause)\s+(\d+(?:\.\d+)*)",
            r"(?:annexe?|appendix)\s+([A-Za-z])"
        ]
        
        # Identify cross-references
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            section_number = metadata.get('section_number', f"unknown_{i}")
            if section_number not in self.graph.nodes:
                continue
                
            # Find all cross-references
            for pattern in ref_patterns:
                for match in re.finditer(pattern, doc, re.IGNORECASE):
                    referenced_section = match.group(1)
                    if referenced_section in self.graph.nodes:
                        self.graph.add_edge(
                            section_number, 
                            referenced_section, 
                            relation="references",
                            weight=0.8
                        )
    
    def retrieve(self, query: str, n_results: int = 5, max_hop: int = 2) -> List[Dict]:
        """
        Retrieve relevant documents using both vector similarity and graph traversal.
        
        Args:
            query: User query
            n_results: Number of results to return
            max_hop: Maximum number of hops in graph traversal
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.graph or not self.graph.nodes:
            print("Graph is empty. Falling back to vector-only search.")
            return self.chroma_manager.search(query, n_results=n_results)
        
        # First, get initial results from vector search
        initial_results = self.chroma_manager.search(query, n_results=n_results)
        
        # Extract node IDs from initial results
        initial_nodes = set()
        for result in initial_results:
            if 'metadata' in result and 'section_number' in result['metadata']:
                node_id = result['metadata']['section_number']
                if node_id in self.graph.nodes:
                    initial_nodes.add(node_id)
        
        # If no nodes from initial results are in the graph, return vector results
        if not initial_nodes:
            return initial_results
        
        # Collect neighboring nodes through graph traversal
        enhanced_nodes = self._graph_traversal(initial_nodes, max_hop)
        
        # Retrieve additional documents based on graph traversal
        additional_docs = []
        for node in enhanced_nodes:
            if node in self.node_to_document:
                doc = self.node_to_document[node]
                # Find the corresponding metadata
                meta = None
                for result in self.chroma_manager.collection.get(
                    where={"section_number": node},
                    include=["metadatas", "documents"]
                )['metadatas']:
                    if meta is None:
                        meta = result
                if doc and meta:
                    additional_docs.append({
                        'document': doc,
                        'metadata': meta,
                        'distance': 1.0,  # Default distance for graph-derived results
                        'source': 'graph'
                    })
        
        # Combine and deduplicate results
        all_results = initial_results + additional_docs
        
        # Remove duplicates while preserving order
        seen = set()
        deduplicated = []
        for result in all_results:
            if 'metadata' in result and 'section_number' in result['metadata']:
                section = result['metadata']['section_number']
                if section not in seen:
                    seen.add(section)
                    deduplicated.append(result)
        
        # Re-rank combined results (could be enhanced with more sophisticated ranking)
        ranked_results = self._rerank_results(query, deduplicated)
        
        return ranked_results[:n_results]
    
    def _graph_traversal(self, initial_nodes: Set[str], max_hop: int = 2) -> Set[str]:
        """
        Perform graph traversal to find related nodes.
        
        Args:
            initial_nodes: Set of starting node IDs
            max_hop: Maximum number of hops in traversal
            
        Returns:
            Set of enhanced node IDs
        """
        enhanced_nodes = set(initial_nodes)
        
        # For each starting node, find neighbors within max_hop distance
        for node in initial_nodes:
            # Get parent nodes (hierarchical context)
            for parent in self.graph.predecessors(node):
                if self.graph.get_edge_data(parent, node).get('relation') != 'previous':
                    enhanced_nodes.add(parent)
            
            # Get child nodes (sub-sections)
            for child in self.graph.successors(node):
                if self.graph.get_edge_data(node, child).get('relation') != 'next':
                    enhanced_nodes.add(child)
            
            # Get sequential neighbors (next/previous sections)
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('relation') in ['next', 'previous']:
                    enhanced_nodes.add(neighbor)
            
            # Get referenced sections
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('relation') == 'references':
                    enhanced_nodes.add(neighbor)
        
        return enhanced_nodes
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results based on combined vector similarity and graph relevance.
        
        Args:
            query: User query
            results: List of retrieved results
            
        Returns:
            Re-ranked list of results
        """
        # Simple re-ranking: vector results first, then graph results
        vector_results = [r for r in results if r.get('source') != 'graph']
        graph_results = [r for r in results if r.get('source') == 'graph']
        
        # Re-rank graph results by their relevance to the query
        if graph_results:
            query_embedding = self.embeddings_manager.embed_text(query)
            for result in graph_results:
                doc_embedding = self.embeddings_manager.embed_text(result['document'])
                similarity = np.dot(query_embedding, doc_embedding)
                result['distance'] = 1.0 - similarity  # Convert similarity to distance
            
            # Sort graph results by distance
            graph_results = sorted(graph_results, key=lambda x: x['distance'])
        
        # Combine results: vector results first, then graph results
        return vector_results + graph_results
    
    def chat_with_graph(self, query: str, n_context: int = 5, max_hop: int = 2) -> Dict:
        """
        Chat with the document using graph-enhanced RAG.
        
        Args:
            query: User's question
            n_context: Number of relevant chunks to use as context
            max_hop: Maximum number of hops in graph traversal
            
        Returns:
            Dictionary with response and sources
        """
        # Get enhanced results from graph-based retrieval
        results = self.retrieve(query, n_results=n_context, max_hop=max_hop)
        
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
        
        return {
            "response": response,
            "sources": results
        }
