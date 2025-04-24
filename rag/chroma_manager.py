import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .embeddings_manager import EmbeddingsManager


class ChromaDBManager:
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        persist_directory: str = "chroma_db",
        collection_name: str = "contracts",
    ):
        """
        Initialize ChromaDB manager with an embeddings manager

        Args:
            embeddings_manager: Instance of EmbeddingsManager for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.embeddings_manager = embeddings_manager
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        chunks: List[Dict],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to ChromaDB with their embeddings

        Args:
            chunks: List of document chunks (text content)
            metadata: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not chunks:
            return

        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embeddings_manager.get_embeddings(texts)

        # Prepare metadata with at least one attribute
        if metadata is None:
            metadata = [{"source": "document"} for _ in chunks]
        else:
            # Ensure each metadata dict has at least one attribute
            metadata = [m if m else {"source": "document"} for m in metadata]

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids
        )

    def search(
        self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of results with documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embeddings_manager.get_embeddings([query])[0]

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return formatted_results

    def delete_collection(self) -> None:
        """Delete the current collection"""
        self.client.delete_collection(self.collection.name)

    def reset(self) -> None:
        """Reset the database"""
        self.client.reset()

    def get_all_documents(self):
        """
        Get all documents from the ChromaDB collection
        
        Returns:
            List of documents with their metadata
        """
        try:
            results = self.collection.get()
            documents = []
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                documents.append({
                    'content': doc,
                    'metadata': metadata,
                    'id': results['ids'][i] if 'ids' in results else f"doc_{i}"
                })
                
            return documents
        except Exception as e:
            print(f"Error retrieving all documents: {str(e)}")
            return []
