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
        "NV-RerankQA-Mistral-4B-v3": "nvidia/rerankqa-mistral-4b-v3",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "Jina-ColBERT-v1": "jinaai/jina-embeddings-v2-base-en"
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
        
    def rerank(self, query: str, documents: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Rerank documents based on their relevance to the query"""
        if not documents:
            return []
            
        # Prepare pairs for reranking
        pairs = []
        for doc in documents:
            pairs.append((query, doc["text"] if isinstance(doc, dict) else doc))
            
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
            scores = torch.sigmoid(scores).cpu().numpy()
            
        # Sort documents by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k:
            scored_docs = scored_docs[:top_k]
            
        # Return reranked documents with scores
        reranked = []
        for doc, score in scored_docs:
            if isinstance(doc, dict):
                doc["rerank_score"] = float(score)
                reranked.append(doc)
            else:
                reranked.append({
                    "text": doc,
                    "rerank_score": float(score)
                })
                
        return reranked 