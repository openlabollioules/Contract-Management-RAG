import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np

from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__file__)

class Reranker:
    """Handles reranking of search results using different models"""
    
    MODELS = {
        #"bge-reranker-large": "BAAI/bge-reranker-large",
        "mxbai-rerank-large-v2": "mixedbread-ai/mxbai-rerank-large-v2"
        # "Jina-ColBERT-v1": "jinaai/jina-embeddings-v2-base-en"
    }
    
    def __init__(self, model_name: str):
        """Initialize reranker with specified model"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.MODELS.keys())}")
            
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        model_path = self.MODELS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        logger.info(f"Initialized reranker with model: {model_name}")
        
    def rerank(self, query: str, documents: List, top_k: Optional[int] = None) -> List:
        """Rerank documents based on their relevance to the query"""
        if not documents:
            logger.warning("No documents provided to rerank, returning empty list")
            return []
            
        # Add debug logging to inspect documents and their structure
        logger.info(f"Reranking {len(documents)} documents")
        logger.debug(f"Documents type: {type(documents)}")
        logger.debug(f"Documents sample: {documents[:1] if len(documents) > 0 else 'empty'}")
            
        # Prepare pairs for reranking
        pairs = []
        for doc in documents:
            if isinstance(doc, dict):
                # Support both 'text' and 'document' keys used in different parts of the system
                if "text" in doc:
                    doc_text = doc["text"]
                elif "document" in doc:
                    doc_text = doc["document"]
                else:
                    # Si aucun champ de texte reconnu, utiliser une représentation string
                    logger.warning(f"Format de document non reconnu: {doc.keys() if hasattr(doc, 'keys') else type(doc)}")
                    doc_text = str(doc)
            else:
                # Si le document est déjà une chaîne
                doc_text = doc
                
            pairs.append((query, doc_text))
            
        # Tokenize pairs
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # Get relevance scores
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze()
            
            # Add logging for scores shape before sigmoid
            logger.debug(f"Scores shape before sigmoid: {scores.shape}")
            scores = torch.sigmoid(scores).cpu().numpy()
            logger.debug(f"Scores shape after sigmoid: {scores.shape}, type: {type(scores)}")
            
        # Sort documents by score
        try:
            scored_docs = list(zip(documents, scores))
            logger.debug(f"Scored docs length: {len(scored_docs) if scored_docs else 'Failed to create'}")
        except Exception as e:
            logger.error(f"Error creating scored_docs: {e}")
            logger.error(f"Documents: {documents}")
            logger.error(f"Scores: {scores}, shape: {scores.shape if hasattr(scores, 'shape') else 'no shape'}")
            # If scores is a scalar, convert to an array with one element
            if not hasattr(scores, "__iter__"):
                logger.info("Converting scalar score to array")
                scores = np.array([scores])
            scored_docs = list(zip(documents, [scores] * len(documents)))
            
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k:
            scored_docs = scored_docs[:top_k]
            
        # Return reranked documents with scores
        reranked = []
        for doc, score in scored_docs:
            if isinstance(doc, dict):
                # Add score but preserve original document format
                doc_copy = doc.copy()  # Create a copy to avoid modifying the original
                doc_copy["rerank_score"] = float(score)
                reranked.append(doc_copy)
            else:
                # If it was a string, wrap in a dict with text field
                reranked.append({
                    "text": doc,
                    "rerank_score": float(score)
                })
                
        return reranked 