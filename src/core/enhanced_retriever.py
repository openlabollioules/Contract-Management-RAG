#!/usr/bin/env python3
"""
enhanced_retriever.py - Extension du retriever standard
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from loguru import logger

class EnhancedRetrieverWrapper:
    """
    Wrapper pour ajouter des fonctionnalités avancées à n'importe quel retriever existant
    """
    def __init__(self, 
                 base_retriever: Any,
                 fallback_enabled: bool = True,
                 threshold_multiplier: float = 0.8):
        """
        Initialise le wrapper de retriever amélioré
        
        Args:
            base_retriever: Le retriever de base à étendre
            fallback_enabled: Si True, utilise un mécanisme de repli en cas d'échec
            threshold_multiplier: Multiplicateur pour le seuil de similarité
        """
        self.base_retriever = base_retriever
        self.fallback_enabled = fallback_enabled
        self.threshold_multiplier = threshold_multiplier
        
        logger.info("Enhanced Retriever initialisé")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """
        Récupère les documents pertinents
        
        Args:
            query: Texte de la requête
            top_k: Nombre de résultats à retourner
            **kwargs: Arguments supplémentaires à passer au retriever de base
            
        Returns:
            Liste de documents pertinents
        """
        results = []
        
        # Essayer plusieurs méthodes pour obtenir des résultats avec le retriever de base
        if hasattr(self.base_retriever, 'retrieve'):
            try:
                results = self.base_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    **kwargs
                )
            except TypeError as e:
                # Le retriever peut avoir une signature différente
                logger.warning(f"Erreur lors de l'appel du retriever de base: {e}")
                try:
                    results = self.base_retriever.retrieve(
                        query=query,
                        k=top_k,
                        **kwargs
                    )
                except Exception as e2:
                    logger.error(f"Deuxième erreur lors de l'appel du retriever: {e2}")
                    results = []
        else:
            # Fallback: appeler directement le retriever comme une fonction
            try:
                results = self.base_retriever(
                    query=query,
                    top_k=top_k,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'appel du retriever de base: {e}")
                results = []
        
        # Vérifier si nous avons obtenu des résultats
        if not results and self.fallback_enabled:
            logger.info("Aucun résultat trouvé, utilisation de la méthode de fallback")
            # Ici, nous pourrions mettre en place une stratégie de fallback comme reformuler la requête
            # ou utiliser une recherche plus générale
            
            # Pour l'instant, on retourne simplement un résultat de fallback
            formatted_fallback = [{
                "content": f"Aucun résultat trouvé pour la requête: {query}. Essayez de reformuler votre question.",
                "metadata": {
                    "is_fallback": True,
                    "source": "fallback_message"
                }
            }]
            
            return formatted_fallback
        
        return results

# Exemple d'utilisation avec LangChain
class LangChainEnhancedRetriever:
    """
    Implémentation spécifique pour LangChain
    """
    def __init__(self, 
                 vector_store,
                 fallback_enabled: bool = True,
                 search_type: str = "similarity"):
        """
        Initialise le retriever amélioré pour LangChain
        
        Args:
            vector_store: Le vector store LangChain (Chroma, FAISS, etc.)
            fallback_enabled: Si True, utilise une stratégie de repli en cas d'échec
            search_type: Type de recherche à utiliser (similarity, mmr)
        """
        self.vector_store = vector_store
        self.fallback_enabled = fallback_enabled
        self.search_type = search_type
        
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Méthode compatible LangChain pour récupérer des documents
        
        Args:
            query: Texte de la requête
            k: Nombre de résultats à retourner
            
        Returns:
            Liste de documents pertinents
        """
        # Effectuer une recherche standard
        if self.search_type == "mmr":
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k
            )
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
        
        # Vérifier si nous avons des résultats
        if not results and self.fallback_enabled:
            logger.info("Aucun résultat trouvé, utilisation du fallback")
            
            # Implémentation d'un fallback simple: message générique
            from langchain.schema import Document
            
            formatted_fallback = [
                Document(
                    page_content=f"Aucun résultat trouvé pour la requête: {query}. Essayez de reformuler votre question.",
                    metadata={
                        "is_fallback": True,
                        "source": "fallback_message"
                    }
                )
            ]
            
            return formatted_fallback
        
        return results

# Fonction utilitaire pour créer facilement un retriever amélioré
def create_enhanced_retriever(retriever_type: str, 
                             vector_store: Any,
                             **kwargs) -> Any:
    """
    Crée un retriever amélioré selon le type spécifié
    
    Args:
        retriever_type: Type de retriever ('standard', 'langchain')
        vector_store: La base vectorielle à utiliser
        **kwargs: Arguments supplémentaires pour le retriever
        
    Returns:
        Un retriever amélioré
    """
    if retriever_type.lower() == "langchain":
        return LangChainEnhancedRetriever(vector_store, **kwargs)
    else:
        # Pour le type standard, nous devons obtenir un retriever de base
        # Cela dépend de l'implémentation de vector_store
        if hasattr(vector_store, 'as_retriever'):
            base_retriever = vector_store.as_retriever()
        else:
            base_retriever = vector_store
            
        return EnhancedRetrieverWrapper(base_retriever, **kwargs)
