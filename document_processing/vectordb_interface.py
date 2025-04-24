import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils.logger import setup_logger

from .text_vectorizer import TextVectorizer

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class VectorDBInterface:
    def __init__(
        self,
        embeddings_manager: TextVectorizer,
        persist_directory: str = "chroma_db",
        collection_name: str = "contracts",
    ):
        """
        Initialize ChromaDB manager with an embeddings manager

        Args:
            embeddings_manager: Instance of TextVectorizer for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.embeddings_manager = embeddings_manager
        self.persist_directory = persist_directory
        logger.info(
            f"Initialisation de VectorDBInterface (persist_directory={persist_directory}, collection={collection_name})"
        )

        # Initialize ChromaDB client
        logger.debug(f"Création du client ChromaDB persistant")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or get collection
        logger.debug(f"Récupération ou création de la collection '{collection_name}'")
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{collection_name}' initialisée")

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
            logger.warning(
                "Tentative d'ajout de documents avec une liste de chunks vide"
            )
            return

        logger.info(f"Ajout de {len(chunks)} documents à ChromaDB")

        # Generate embeddings
        logger.debug("Génération des embeddings pour les chunks")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embeddings_manager.get_embeddings(texts)
        logger.debug(f"Embeddings générés: {len(embeddings)}")

        # Prepare metadata with at least one attribute
        if metadata is None:
            logger.debug(
                "Aucune métadonnée fournie, utilisation de métadonnées par défaut"
            )
            metadata = [{"source": "document"} for _ in chunks]
        else:
            # Ensure each metadata dict has at least one attribute
            logger.debug("Préparation des métadonnées")
            metadata = [m if m else {"source": "document"} for m in metadata]

        # Generate IDs if not provided
        if ids is None:
            logger.debug("Génération d'IDs UUID pour les documents")
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Add to ChromaDB
        logger.debug("Ajout des documents à ChromaDB")
        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids
        )
        logger.info(
            f"Documents ajoutés avec succès (collection: {self.collection.name})"
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
        logger.info(
            f"Recherche dans ChromaDB: '{query}' (n_results={n_results}, filtres={filter_metadata})"
        )

        # Generate query embedding
        logger.debug("Génération de l'embedding pour la requête")
        query_embedding = self.embeddings_manager.get_embeddings([query])[0]

        # Search in ChromaDB
        logger.debug(f"Requête dans la collection '{self.collection.name}'")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
        )

        # Format results
        logger.debug(
            f"Formatage des résultats ({len(results['ids'][0])} documents trouvés)"
        )
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

        logger.info(f"Recherche terminée, {len(formatted_results)} résultats")
        return formatted_results

    def delete_collection(self) -> None:
        """Delete the current collection"""
        logger.warning(f"Suppression de la collection '{self.collection.name}'")
        self.client.delete_collection(self.collection.name)
        logger.info(f"Collection '{self.collection.name}' supprimée")

    def reset(self) -> None:
        """Reset the database"""
        logger.warning("Réinitialisation complète de la base de données ChromaDB")
        self.client.reset()
        logger.info("Base de données réinitialisée")
