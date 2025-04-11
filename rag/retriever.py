from typing import List, Optional

import faiss
import numpy as np
import torch


class Retriever:
    def __init__(self, dimension: int = 1024):
        """Initialize the retriever with FAISS index"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(
            dimension
        )  # Inner product index (cosine similarity)
        self.texts = []

    def add_texts(self, texts: List[str], embeddings: List[np.ndarray]):
        """Add texts and their embeddings to the index"""
        if not texts or not embeddings:
            return

        # Convert embeddings to numpy array if needed
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = [emb.cpu().numpy() for emb in embeddings]

        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)

        # Add to index
        self.index.add(embeddings_array)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Search for most similar texts given a query embedding"""
        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return relevant texts
        return [self.texts[idx] for idx in indices[0]]


# Global retriever instance
_retriever: Optional[Retriever] = None


def add_to_index(texts: List[str], embeddings: List[np.ndarray]):
    """Add texts and embeddings to the global retriever"""
    global _retriever

    if _retriever is None:
        # Initialize with dimension from first embedding
        dim = (
            embeddings[0].shape[0]
            if isinstance(embeddings[0], np.ndarray)
            else embeddings[0].size
        )
        _retriever = Retriever(dimension=dim)

    _retriever.add_texts(texts, embeddings)


def query_index(query_embedding: List[np.ndarray], top_k: int = 5) -> List[str]:
    """Query the global retriever"""
    global _retriever

    if _retriever is None:
        raise RuntimeError("Index not initialized. Call add_to_index first.")

    return _retriever.search(query_embedding[0], top_k)
