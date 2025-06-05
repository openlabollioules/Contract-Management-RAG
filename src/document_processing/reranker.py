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
        # --- Correction gestion device pour MPS (Mac GPU) ---
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device for reranker (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device for reranker")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for reranker")
        
        # Load model and tokenizer
        model_path = self.MODELS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # --- Vérification compatibilité MPS ---
        if self.device.type == "mps":
            try:
                _ = torch.ones(1).to(self.device)
            except Exception as e:
                logger.warning(f"MPS device detected but not fully supported: {e}")
                logger.warning("Certaines opérations peuvent échouer sur MPS. Essayez CPU si problème.")
        
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
            
        # Process one document at a time to avoid batch processing issues
        scores = []
        for pair in pairs:
            # --- Correction: Tokenization pour modèles sequence classification (text, text_pair) ---
            features = self.tokenizer(
                text=pair[0],
                text_pair=pair[1],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            features = {k: v.to(self.device) for k, v in features.items()}
            
            # Get relevance score for this pair
            with torch.no_grad():
                score = self.model(**features).logits.squeeze()
                # Si score est un tenseur avec plusieurs éléments, prendre le premier
                if hasattr(score, "shape") and score.shape and score.shape[0] > 1:
                    score = score[0]
                score = torch.sigmoid(score).cpu().numpy()
                # Si c'est toujours un tableau numpy avec plusieurs éléments, prendre le premier
                if isinstance(score, np.ndarray) and score.size > 1:
                    score = score[0]
                scores.append(float(score))
            
        # Add logging for scores
        logger.debug(f"Processed {len(scores)} documents with individual scoring")
            
        # Sort documents by score
        try:
            scored_docs = list(zip(documents, scores))
            logger.debug(f"Scored docs length: {len(scored_docs) if scored_docs else 'Failed to create'}")
        except Exception as e:
            logger.error(f"Error creating scored_docs: {e}")
            logger.error(f"Documents: {documents}")
            logger.error(f"Scores: {scores}")
            # Fallback to returning the original documents without reranking
            return documents
        
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