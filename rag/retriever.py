from typing import List, Optional

import faiss
import numpy as np
import torch

from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class Retriever:
    def __init__(self, dimension: int = 1024):
        """Initialize the retriever with FAISS index"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(
            dimension
        )  # Inner product index (cosine similarity)
        self.texts = []
        logger.info(f"Retriever initialisé avec dimension={dimension}")
        logger.debug("Index FAISS (IndexFlatIP) créé")

    def add_texts(self, texts: List[str], embeddings: List[np.ndarray]):
        """Add texts and their embeddings to the index"""
        if not texts or not embeddings:
            logger.warning("Tentative d'ajout de textes vides ou d'embeddings vides")
            return

        logger.info(f"Ajout de {len(texts)} textes à l'index FAISS")

        # Convert embeddings to numpy array if needed
        if isinstance(embeddings[0], torch.Tensor):
            logger.debug("Conversion des embeddings torch.Tensor en numpy.array")
            embeddings = [emb.cpu().numpy() for emb in embeddings]

        # Normalize embeddings for cosine similarity
        logger.debug("Normalisation des embeddings pour la similarité cosinus")
        embeddings_array = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)

        # Add to index
        logger.debug(f"Ajout de {len(embeddings_array)} embeddings à l'index")
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        logger.info(f"Index mis à jour, contient maintenant {len(self.texts)} textes")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Search for most similar texts given a query embedding"""
        logger.info(f"Recherche des {top_k} textes les plus similaires")

        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            logger.debug(
                "Conversion de l'embedding de requête torch.Tensor en numpy.array"
            )
            query_embedding = query_embedding.cpu().numpy()

        logger.debug("Normalisation de l'embedding de requête")
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        logger.debug(
            f"Recherche dans l'index FAISS (dimension: {self.dimension}, taille: {len(self.texts)})"
        )
        scores, indices = self.index.search(query_embedding, top_k)
        logger.debug(f"Scores de similarité: {scores}")

        # Return relevant texts
        result = [self.texts[idx] for idx in indices[0]]
        logger.info(f"Recherche terminée, {len(result)} textes pertinents trouvés")
        return result


# Global retriever instance
_retriever: Optional[Retriever] = None


def add_to_index(texts: List[str], embeddings: List[np.ndarray]):
    """Add texts and embeddings to the global retriever"""
    global _retriever
    logger.info(f"Ajout de {len(texts)} textes à l'index global")

    if _retriever is None:
        # Initialize with dimension from first embedding
        dim = (
            embeddings[0].shape[0]
            if isinstance(embeddings[0], np.ndarray)
            else embeddings[0].size
        )
        logger.debug(f"Initialisation du Retriever global avec dimension={dim}")
        _retriever = Retriever(dimension=dim)

    logger.debug("Ajout des textes et embeddings à l'instance globale")
    _retriever.add_texts(texts, embeddings)


def query_index(query_embedding: List[np.ndarray], top_k: int = 5) -> List[str]:
    """Query the global retriever"""
    global _retriever
    logger.info(f"Requête sur l'index global (top_k={top_k})")

    if _retriever is None:
        logger.error("Index non initialisé. Appeler add_to_index d'abord.")
        raise RuntimeError("Index not initialized. Call add_to_index first.")

    logger.debug("Exécution de la recherche sur l'instance globale")
    result = _retriever.search(query_embedding[0], top_k)
    logger.info(f"Requête terminée, {len(result)} résultats")
    return result
