import networkx as nx
from typing import List, Dict

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, metadata_list: List[Dict]) -> nx.DiGraph:
        """
        Build a directed graph from the list of metadata.

        Args:
            metadata_list: List of metadata dictionaries for each chunk

        Returns:
            A NetworkX DiGraph object representing the hierarchical structure
        """
        print("Building graph...")

        # Add nodes and edges based on hierarchy
        for metadata in metadata_list:
            section_number = metadata.get("section_number", "unknown")
            hierarchy = metadata.get("hierarchy", ["unknown"])
            document_title = metadata.get("document_title", "unknown")
            
            # Create a node for the section
            self.graph.add_node(section_number, hierarchy=hierarchy, document_title=document_title)

            # Add edges based on parent-child relationships
            if len(hierarchy) > 1:
                parent_section = hierarchy[-2]
                self.graph.add_edge(parent_section, section_number)
        
        print("Graph built successfully.")
        return self.graph

    def display_graph(self):
        """
        Display the graph structure (for debugging purposes).
        """
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.graph)
        labels = {node: node for node in self.graph.nodes()}
        nx.draw(self.graph, pos, with_labels=True, labels=labels, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
        plt.title("Hierarchical Graph of Document Sections")
        plt.show()