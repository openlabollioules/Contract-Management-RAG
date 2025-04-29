import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from rag.ollama_chat import OllamaChat
from rag.chroma_manager import ChromaDBManager
from rag.embeddings_manager import EmbeddingsManager
try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("Warning: leidenalg or igraph not installed. Community detection will be limited.")


class GraphNode:
    def __init__(self, id: str, content: str, metadata: Dict):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.community = None  # Will store community ID when detected
        
    def __repr__(self):
        return f"Node({self.id})"


class GraphEdge:
    def __init__(self, source: str, target: str, relation_type: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight
        
    def __repr__(self):
        return f"Edge({self.source} --[{self.relation_type}]--> {self.target})"


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.communities: Dict[int, List[str]] = {}  # community_id -> list of node_ids
        self.community_summaries: Dict[int, str] = {}  # community_id -> summary text
        
    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        # Remove content from metadata to avoid duplicate keyword argument
        node_metadata = node.metadata.copy()
        if 'content' in node_metadata:
            del node_metadata['content']
        
        # Add community info if available
        if hasattr(node, 'community') and node.community is not None:
            self.graph.add_node(node.id, content=node.content, community=node.community, **node_metadata)
        else:
            self.graph.add_node(node.id, content=node.content, **node_metadata)
        
    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source in self.nodes and edge.target in self.nodes:
            self.edges.append(edge)
            self.graph.add_edge(
                edge.source, 
                edge.target, 
                relation_type=edge.relation_type, 
                weight=edge.weight
            )
    
    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None) -> List[GraphNode]:
        """Get nodes related to the given node with optional filter by relation type"""
        related_nodes = []
        
        if node_id not in self.nodes:
            return []
            
        for edge in self.edges:
            if edge.source == node_id and (relation_type is None or edge.relation_type == relation_type):
                related_nodes.append(self.nodes[edge.target])
                
        return related_nodes
    
    def set_node_community(self, node_id: str, community_id: int) -> None:
        """Set community for a node and update communities dict"""
        if node_id in self.nodes:
            self.nodes[node_id].community = community_id
            
            # Update graph node attribute
            self.graph.nodes[node_id]['community'] = community_id
            
            # Update communities collection
            if community_id not in self.communities:
                self.communities[community_id] = []
            
            if node_id not in self.communities[community_id]:
                self.communities[community_id].append(node_id)
    
    def set_community_summary(self, community_id: int, summary: str) -> None:
        """Store summary for a community"""
        self.community_summaries[community_id] = summary
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
        
    @property
    def edge_count(self) -> int:
        return len(self.edges)


class GraphManager:
    def __init__(
        self, 
        chroma_manager: ChromaDBManager, 
        embeddings_manager: EmbeddingsManager,
        llm_model: str = "mistral-small3.1:latest",
        similarity_threshold: float = 0.85
    ):
        self.chroma_manager = chroma_manager
        self.embeddings_manager = embeddings_manager
        self.llm = OllamaChat(model=llm_model)
        self.similarity_threshold = similarity_threshold
        
    def build_graph(self, chunks: List[Dict]) -> KnowledgeGraph:
        """Build a knowledge graph from the given chunks with community detection"""
        print("\nBuilding enhanced knowledge graph with community detection...")
        
        graph = KnowledgeGraph()
        
        # 1. Add nodes for each chunk
        print("1/6: Adding document chunks to graph...")
        self._add_chunk_nodes(graph, chunks)
        
        # 2. Find semantic similarity edges between nodes
        print("2/6: Identifying semantic similarities between chunks...")
        self._add_similarity_edges(graph)
        
        # 3. Generate relationship edges using LLM
        print("3/6: Generating semantic relationships between chunks...")
        self._add_llm_relationship_edges(graph)
        
        # 4. Add hierarchical structure edges from metadata
        print("4/6: Adding structural relationships from document hierarchy...")
        self._add_hierarchical_edges(graph)
        
        # 5. Perform community detection using Leiden algorithm
        print("5/6: Performing community detection with Leiden algorithm...")
        self._detect_communities(graph)
        
        # 6. Generate summaries for each community
        print("6/6: Generating summaries for each community...")
        self._generate_community_summaries(graph)
        
        print(f"✅ Enhanced graph created with {graph.node_count} nodes, {graph.edge_count} edges, and {len(graph.communities)} communities")
        return graph
    
    def _add_chunk_nodes(self, graph: KnowledgeGraph, chunks: List[Dict]) -> None:
        """Add nodes for each chunk"""
        for i, chunk in enumerate(chunks):
            node_id = f"node_{i}"
            node = GraphNode(
                id=node_id,
                content=chunk["content"],
                metadata=chunk["metadata"]
            )
            graph.add_node(node)
    
    def _add_similarity_edges(self, graph: KnowledgeGraph) -> None:
        """Add edges based on embedding similarity between nodes with filtering"""
        # Get all node contents and IDs
        node_ids = list(graph.nodes.keys())
        node_contents = [node.content for node in graph.nodes.values()]
        
        # Get embeddings for all nodes
        embeddings = self.embeddings_manager.get_embeddings(node_contents)
        
        # Create a priority queue of similarities
        all_similarities = []
        for i, node_id in enumerate(node_ids):
            # Only compare with nodes that have meaningful connections
            # Avoid excessive connections by limiting to top matches
            top_matches_per_node = 3  # Limit connections to most relevant matches
            
            node_similarities = []
            for j, other_id in enumerate(node_ids):
                if i != j:
                    similarity = self.embeddings_manager.compute_similarity(
                        embeddings[i], embeddings[j]
                    )
                    if similarity > self.similarity_threshold:
                        node_similarities.append((similarity, other_id))
            
            # Sort by similarity descending and take only top matches
            node_similarities.sort(reverse=True)
            for similarity, other_id in node_similarities[:top_matches_per_node]:
                edge = GraphEdge(
                    source=node_id,
                    target=other_id,
                    relation_type="similar_to",
                    weight=similarity
                )
                graph.add_edge(edge)
    
    def _add_llm_relationship_edges(self, graph: KnowledgeGraph) -> None:
        """Add edges based on LLM-generated relationships between nodes"""
        processed_pairs: Set[Tuple[str, str]] = set()
        
        # Process nodes in a reasonable batch to avoid too many LLM calls
        for source_id, source_node in list(graph.nodes.items())[:20]:  # Limit initial processing
            # Find semantically similar nodes to analyze relationships
            similar_nodes = []
            for edge in graph.edges:
                if edge.source == source_id and edge.relation_type == "similar_to":
                    similar_nodes.append(graph.nodes[edge.target])
            
            for target_node in similar_nodes[:5]:  # Limit to 5 similar nodes per source
                target_id = target_node.id
                pair_key = tuple(sorted([source_id, target_id]))
                
                # Skip if this pair has been processed
                if pair_key in processed_pairs:
                    continue
                    
                processed_pairs.add(pair_key)
                
                # Get relationship between nodes using LLM
                relationship = self._generate_relationship(source_node, target_node)
                
                if relationship:
                    edge = GraphEdge(
                        source=source_id,
                        target=target_id,
                        relation_type=relationship,
                        weight=1.0
                    )
                    graph.add_edge(edge)
    
    def _generate_relationship(self, source_node: GraphNode, target_node: GraphNode) -> Optional[str]:
        """Generate relationship between two nodes using LLM with improved prompt"""
        prompt = f"""Analyze the semantic relationship between these two text passages from a legal document.
Only identify a relationship if there is a clear, meaningful connection between them.
Express the relationship as a specific predicate (e.g., "defines", "contradicts", "elaborates", "references").
Focus on identifying the strongest and most precise relationship.
Respond only with "no_relation" if the connection is weak or tangential.

PASSAGE 1:
{source_node.content[:500]}

PASSAGE 2:
{target_node.content[:500]}

What is the SINGLE most specific and meaningful relationship from Passage 1 to Passage 2?
Relationship (single word or short phrase only):"""

        # Get relationship from LLM
        response = self.llm.generate(prompt)
        
        # Clean and validate the response
        relationship = response.strip().lower()
        
        # Filter out non-relationship responses
        if relationship in ["no_relation", "none", "n/a"]:
            return None
            
        # Clean up relationship to ensure it's a simple predicate
        relationship = relationship.split()[0] if len(relationship.split()) > 1 else relationship
        relationship = relationship.replace('"', '').replace("'", "")
        
        return relationship
    
    def _add_hierarchical_edges(self, graph: KnowledgeGraph) -> None:
        """Add edges based on document hierarchy from metadata"""
        # Group nodes by hierarchy
        hierarchy_map = {}
        
        for node_id, node in graph.nodes.items():
            section = node.metadata.get("section_number", "unknown")
            if section != "unknown":
                if section not in hierarchy_map:
                    hierarchy_map[section] = []
                hierarchy_map[section].append(node_id)
        
        # Create parent-child relationships based on section numbering
        for section, node_ids in hierarchy_map.items():
            # Find potential parent sections
            parent_section = ".".join(section.split(".")[:-1]) if "." in section else None
            
            if parent_section and parent_section in hierarchy_map:
                for parent_id in hierarchy_map[parent_section]:
                    for child_id in node_ids:
                        edge = GraphEdge(
                            source=parent_id,
                            target=child_id,
                            relation_type="contains",
                            weight=1.0
                        )
                        graph.add_edge(edge)

    def _detect_communities(self, graph: KnowledgeGraph) -> None:
        """
        Detect communities in the graph using the Leiden algorithm for graph clustering
        This identifies groups of closely related nodes that form thematic clusters
        """
        if not LEIDEN_AVAILABLE:
            print("Warning: Leiden algorithm not available. Skipping community detection.")
            return
        
        if graph.node_count == 0:
            print("Graph has no nodes. Skipping community detection.")
            return
        
        try:
            # Convert NetworkX graph to igraph for Leiden algorithm
            # Use undirected version for community detection
            nx_graph = nx.Graph(graph.graph)  # Convert to undirected
            
            # Create igraph from networkx
            g = ig.Graph.from_networkx(nx_graph)
            
            # Get edge weights if available
            weights = []
            for edge in nx_graph.edges():
                weight = nx_graph.edges[edge].get('weight', 1.0)
                weights.append(weight)
            
            # Run Leiden algorithm
            partition = la.find_partition(
                g, 
                la.ModularityVertexPartition, 
                weights=weights,
                resolution_parameter=1.0  # Adjust for granularity (higher = more communities)
            )
            
            # Store detected communities
            for community_id, members in enumerate(partition):
                for node_idx in members:
                    # Convert igraph vertex index to networkx node id
                    node_id = list(graph.nodes.keys())[node_idx]
                    graph.set_node_community(node_id, community_id)
                    
            print(f"✅ Detected {len(partition)} communities")
            
        except Exception as e:
            print(f"Error during community detection: {str(e)}")
    
    def _generate_community_summaries(self, graph: KnowledgeGraph) -> None:
        """
        Generate summaries for each community using LLM
        This helps understand the key themes in each detected community
        """
        if not graph.communities:
            print("No communities detected. Skipping summary generation.")
            return
        
        for community_id, node_ids in graph.communities.items():
            # Skip if community is too small
            if len(node_ids) < 2:
                graph.set_community_summary(community_id, "Single node community - no summary required")
                continue
                
            # Gather content from this community
            community_content = []
            for node_id in node_ids[:5]:  # Limit to 5 nodes to avoid huge prompts
                node = graph.nodes[node_id]
                # Get a preview of each node's content
                content_preview = node.content.replace('\n', ' ')[:300]  # First 300 chars
                community_content.append(content_preview)
            
            # Build prompt for summarization
            prompt = f"""Summarize the main theme or topic of these related text passages from legal or contract documents.
Create a concise, informative summary (1-2 sentences) that captures the key concept these passages share.

RELATED PASSAGES:
{' '.join(community_content)}

SUMMARY:"""
            
            try:
                # Generate summary
                summary = self.llm.generate(prompt)
                graph.set_community_summary(community_id, summary.strip())
            except Exception as e:
                print(f"Error generating summary for community {community_id}: {str(e)}")
                graph.set_community_summary(community_id, "Summary generation failed")
                
        print(f"✅ Generated summaries for {len(graph.communities)} communities")

    def visualize_graph(self, graph: KnowledgeGraph, output_path: str = "knowledge_graph.png", 
                       max_nodes: int = 50, title: str = "Contract Knowledge Graph") -> None:
        """
        Generate and save a visual representation of the knowledge graph.
        
        Args:
            graph: The knowledge graph to visualize
            output_path: Path to save the image file
            max_nodes: Maximum number of nodes to include in visualization
            title: Title for the graph visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import random
        
        # Create a smaller subgraph if the graph is too large
        if len(graph.nodes) > max_nodes:
            print(f"Graph too large ({len(graph.nodes)} nodes), showing only {max_nodes} nodes")
            # Select the first max_nodes nodes for visualization
            node_subset = list(graph.nodes.keys())[:max_nodes]
            # Create a subgraph with only these nodes
            subgraph = nx.DiGraph()
            for node_id in node_subset:
                node = graph.nodes[node_id]
                node_attrs = graph.graph.nodes[node_id]
                subgraph.add_node(node_id, **node_attrs)
            
            # Add edges between the selected nodes
            for edge in graph.edges:
                if edge.source in node_subset and edge.target in node_subset:
                    subgraph.add_edge(
                        edge.source, edge.target, 
                        relation_type=edge.relation_type, 
                        weight=edge.weight
                    )
            viz_graph = subgraph
        else:
            viz_graph = graph.graph
        
        plt.figure(figsize=(14, 10))
        plt.title(title, fontsize=16)
        
        # Create a position layout that respects communities
        pos = nx.spring_layout(viz_graph, seed=42, k=0.5)
        
        # Get community information
        node_colors = []
        node_sizes = []
        communities_present = False
        
        # Create colormap for communities
        cmap = plt.cm.get_cmap('tab20', 20)  # 20 distinct colors
        
        for node in viz_graph.nodes():
            # Check if we have community info
            community = viz_graph.nodes[node].get('community')
            if community is not None:
                communities_present = True
                color = cmap(community % 20)  # Wrap around for more than 20 communities
                
                # Node size based partly on community size
                comm_size = len(graph.communities.get(community, []))
                size = 300 + (comm_size * 5)
                
            else:
                color = "skyblue"
                size = 300
                
            node_colors.append(color)
            node_sizes.append(size)
        
        # Define colors for different relation types
        relation_types = set()
        for _, _, data in viz_graph.edges(data=True):
            relation_type = data.get('relation_type', 'unknown')
            relation_types.add(relation_type)
        
        # Create a color map for relations
        colors = list(mcolors.TABLEAU_COLORS.values())
        random.seed(42)  # For reproducibility
        relation_colors = {rel: colors[i % len(colors)] for i, rel in enumerate(relation_types)}
        
        # Draw nodes
        nx.draw_networkx_nodes(
            viz_graph, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.7
        )
        
        # Draw edges with colors based on relation type
        for relation in relation_types:
            edge_list = [(u, v) for u, v, d in viz_graph.edges(data=True) 
                        if d.get('relation_type') == relation]
            if edge_list:
                nx.draw_networkx_edges(
                    viz_graph, pos,
                    edgelist=edge_list,
                    width=1.5,
                    alpha=0.7,
                    edge_color=relation_colors[relation],
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=10
                )
        
        # Draw node labels (section numbers or metadata)
        labels = {}
        for node_id in viz_graph.nodes():
            node_data = viz_graph.nodes[node_id]
            if 'section_number' in node_data:
                labels[node_id] = node_data['section_number']
            else:
                labels[node_id] = node_id
        
        nx.draw_networkx_labels(
            viz_graph, pos,
            labels=labels,
            font_size=8
        )
        
        # Add a legend for relation types
        legend_elements = [plt.Line2D([0], [0], color=relation_colors[rel], lw=2, label=rel) 
                          for rel in relation_types]
        
        # Add community info to legend if present
        if communities_present:
            # Add a title for communities section in legend
            plt.legend(handles=legend_elements, loc='upper right', title="Relation Types")
            
            # Generate community summary report
            self._generate_community_report(graph, output_path.replace('.png', '_communities.txt'))
        else:
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Remove axis
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graph visualization saved to {output_path}")
        
        return output_path

    def _generate_community_report(self, graph: KnowledgeGraph, output_path: str) -> None:
        """
        Generate a report of communities with their summaries and key nodes
        """
        if not graph.communities:
            return
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# CONTRACT KNOWLEDGE GRAPH - COMMUNITY ANALYSIS\n\n")
            
            # Sort communities by size
            communities = sorted(
                graph.communities.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            for community_id, node_ids in communities:
                f.write(f"## Community {community_id} ({len(node_ids)} nodes)\n\n")
                
                # Write summary
                if community_id in graph.community_summaries:
                    f.write(f"**Summary**: {graph.community_summaries[community_id]}\n\n")
                
                # List key nodes (limit to 5)
                f.write("**Key Nodes**:\n\n")
                for node_id in node_ids[:5]:
                    node = graph.nodes[node_id]
                    
                    # Extract metadata
                    section = node.metadata.get('section_number', 'N/A')
                    doc_title = node.metadata.get('document_title', 'N/A')
                    
                    # Create content preview
                    content = node.content.replace('\n', ' ')
                    preview = content[:100] + '...' if len(content) > 100 else content
                    
                    f.write(f"- **Node {node_id}** (Section: {section}, Document: {doc_title})\n")
                    f.write(f"  Preview: {preview}\n\n")
                
                f.write("\n")
            
        print(f"✅ Community report saved to {output_path}")

    def export_graph_details(self, graph: KnowledgeGraph, output_path: str = "graph_details.txt") -> None:
        """
        Export detailed information about all nodes and relations in the graph to a text file.
        
        Args:
            graph: The knowledge graph to export
            output_path: Path to save the text file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# KNOWLEDGE GRAPH DETAILS\n\n")
            f.write(f"Total Nodes: {graph.node_count}\n")
            f.write(f"Total Relations: {graph.edge_count}\n\n")
            
            # Write community information if available
            if graph.communities:
                f.write("## COMMUNITIES\n\n")
                f.write(f"Total Communities: {len(graph.communities)}\n\n")
                
                for community_id, node_ids in graph.communities.items():
                    f.write(f"### Community {community_id}\n")
                    f.write(f"Size: {len(node_ids)} nodes\n")
                    
                    # Write summary if available
                    if community_id in graph.community_summaries:
                        f.write(f"Summary: {graph.community_summaries[community_id]}\n")
                    
                    f.write("\n")
            
            # Write node details
            f.write("## NODES\n\n")
            for i, (node_id, node) in enumerate(graph.nodes.items(), 1):
                f.write(f"### Node {i}: {node_id}\n")
                
                # Write community if available
                if hasattr(node, 'community') and node.community is not None:
                    f.write(f"Community: {node.community}\n")
                
                f.write(f"Section: {node.metadata.get('section_number', 'N/A')}\n")
                
                # Write hierarchy if available
                hierarchy = node.metadata.get('hierarchy', [])
                if hierarchy and isinstance(hierarchy, list):
                    f.write(f"Hierarchy: {' -> '.join(hierarchy)}\n")
                elif hierarchy:
                    f.write(f"Hierarchy: {hierarchy}\n")
                    
                # Write other metadata
                f.write("Metadata:\n")
                for key, value in node.metadata.items():
                    if key not in ['section_number', 'hierarchy', 'content']:
                        # Handle complex values (like lists or dicts)
                        if isinstance(value, (list, dict)):
                            value = str(value)
                        f.write(f"  - {key}: {value}\n")
                
                # Write content preview
                content_preview = node.content.replace('\n', ' ')[:200]
                f.write(f"Content Preview: {content_preview}...\n\n")
            
            # Write edge details
            f.write("## RELATIONS\n\n")
            
            # Group edges by relation type for better organization
            relation_groups = {}
            for edge in graph.edges:
                if edge.relation_type not in relation_groups:
                    relation_groups[edge.relation_type] = []
                relation_groups[edge.relation_type].append(edge)
            
            # Write edges grouped by relation type
            for relation_type, edges in relation_groups.items():
                f.write(f"### Relation Type: {relation_type}\n")
                f.write(f"Count: {len(edges)}\n\n")
                
                for i, edge in enumerate(edges[:10], 1):  # Limit to 10 examples per type
                    source_node = graph.nodes[edge.source]
                    target_node = graph.nodes[edge.target]
                    
                    f.write(f"#### Relation {i}\n")
                    f.write(f"Source Node: {edge.source}\n")
                    
                    # Write community of source node if available
                    if hasattr(source_node, 'community') and source_node.community is not None:
                        f.write(f"  - Community: {source_node.community}\n")
                        
                    f.write(f"  - Section: {source_node.metadata.get('section_number', 'N/A')}\n")
                    source_preview = source_node.content.replace('\n', ' ')[:100]
                    f.write(f"  - Content: {source_preview}...\n")
                    
                    f.write(f"Target Node: {edge.target}\n")
                    
                    # Write community of target node if available
                    if hasattr(target_node, 'community') and target_node.community is not None:
                        f.write(f"  - Community: {target_node.community}\n")
                        
                    f.write(f"  - Section: {target_node.metadata.get('section_number', 'N/A')}\n")
                    target_preview = target_node.content.replace('\n', ' ')[:100]
                    f.write(f"  - Content: {target_preview}...\n")
                    
                    f.write(f"Weight: {edge.weight}\n\n")
                
                # If there are more edges than shown
                if len(edges) > 10:
                    f.write(f"... and {len(edges) - 10} more {relation_type} relations\n")
                
                f.write("\n")
            
            # Write statistics about communities and relations
            f.write("## STATISTICS\n\n")
            
            if graph.communities:
                f.write("### Community Statistics\n\n")
                community_sizes = [len(nodes) for nodes in graph.communities.values()]
                
                f.write(f"Total Communities: {len(graph.communities)}\n")
                f.write(f"Average Community Size: {sum(community_sizes)/len(community_sizes):.1f} nodes\n")
                f.write(f"Largest Community: {max(community_sizes)} nodes\n")
                f.write(f"Smallest Community: {min(community_sizes)} nodes\n\n")
            
            f.write("### Relation Statistics\n\n")
            for relation_type, edges in relation_groups.items():
                f.write(f"{relation_type}: {len(edges)} relations\n")
        
        print(f"✅ Graph details exported to {output_path}")
