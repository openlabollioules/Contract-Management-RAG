#!/usr/bin/env python3
"""
enhanced_retriever.py - Extension du retriever avec préfiltrage par idG
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import chromadb
from loguru import logger

# Importer le module MatrixProcessor
from src.core.matrix_processor import MatrixProcessor

class EnhancedRetrieverWrapper:
    """
    Wrapper pour ajouter le préfiltrage basé sur la Matrice G à n'importe quel retriever existant
    """
    def __init__(self, 
                 base_retriever: Any,
                 matrix_processor: MatrixProcessor,
                 fallback_enabled: bool = True,
                 metadata_field: str = "idG",
                 threshold_multiplier: float = 0.8):
        """
        Initialise le wrapper de retriever amélioré
        
        Args:
            base_retriever: Le retriever de base à étendre
            matrix_processor: Instance de MatrixProcessor
            fallback_enabled: Si True, utilise les résumés de la Matrice G en fallback
            metadata_field: Nom du champ de métadonnées contenant l'idG
            threshold_multiplier: Multiplicateur pour le seuil de similarité lors de la détection d'idG
        """
        self.base_retriever = base_retriever
        self.matrix_processor = matrix_processor
        self.fallback_enabled = fallback_enabled
        self.metadata_field = metadata_field
        self.threshold_multiplier = threshold_multiplier
        
        logger.info("Enhanced Retriever initialisé")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """
        Récupère les documents pertinents avec préfiltrage par idG
        
        Args:
            query: Texte de la requête
            top_k: Nombre de résultats à retourner
            **kwargs: Arguments supplémentaires à passer au retriever de base
            
        Returns:
            Liste de documents pertinents
        """
        # 1. Détecter les idG pertinents pour la question
        idg_list = self.matrix_processor.detect_idg_from_query(query)
        logger.info(f"idG détectés pour la requête '{query}': {idg_list}")
        
        # 2. Si des idG ont été détectés, ajouter un filtre sur les métadonnées
        results = []
        
        if idg_list:
            # Construire le filtre selon l'interface du retriever de base
            # Nous gérons plusieurs cas possibles
            if hasattr(self.base_retriever, 'retrieve_with_filter'):
                # Cas d'un retriever avec méthode de filtrage dédiée
                filter_dict = {self.metadata_field: {"$in": idg_list}}
                results = self.base_retriever.retrieve_with_filter(
                    query=query, 
                    filter=filter_dict,
                    top_k=top_k,
                    **kwargs
                )
            elif hasattr(self.base_retriever, 'retrieve'):
                # Essayer de passer le filtre comme argument nommé
                try:
                    filter_dict = {self.metadata_field: {"$in": idg_list}}
                    results = self.base_retriever.retrieve(
                        query=query,
                        filter=filter_dict,
                        top_k=top_k,
                        **kwargs
                    )
                except TypeError:
                    # Si le retriever n'accepte pas de filtre, utiliser la méthode standard
                    logger.warning("Le retriever de base ne supporte pas le filtrage par métadonnées")
                    results = self.base_retriever.retrieve(
                        query=query,
                        top_k=top_k,
                        **kwargs
                    )
                    
                    # Filtrer manuellement les résultats après récupération
                    if results:
                        filtered_results = []
                        for doc in results:
                            if 'metadata' in doc and self.metadata_field in doc['metadata']:
                                if doc['metadata'][self.metadata_field] in idg_list:
                                    filtered_results.append(doc)
                        
                        if filtered_results:
                            results = filtered_results[:top_k]
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
        else:
            # Si aucun idG n'est détecté, utiliser le retriever de base sans filtre
            logger.info("Aucun idG détecté, utilisation du retriever sans filtre")
            
            if hasattr(self.base_retriever, 'retrieve'):
                results = self.base_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    **kwargs
                )
            else:
                try:
                    results = self.base_retriever(
                        query=query,
                        top_k=top_k,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Erreur lors de l'appel du retriever de base: {e}")
                    results = []
        
        # 3. Fallback: si aucun résultat n'est trouvé mais des idG sont détectés
        if not results and idg_list and self.fallback_enabled:
            logger.info("Aucun résultat trouvé, utilisation du fallback avec les résumés de la Matrice G")
            fallback_results = self.matrix_processor.get_summaries_from_matrix_g(idg_list)
            
            # Transformer les résultats de fallback pour qu'ils correspondent au format attendu
            formatted_fallback = []
            for fb in fallback_results:
                formatted_result = {
                    "content": f"Résumé canonique: {fb['titre']} - {fb['résumé']}",
                    "metadata": {
                        "idG": fb["idG"],
                        "titre": fb["titre"],
                        "type": fb["type"],
                        "is_fallback": True,
                        "source": "matrix_g_fallback"
                    }
                }
                formatted_fallback.append(formatted_result)
            
            return formatted_fallback
        
        return results

# Exemple d'utilisation avec LangChain
class LangChainEnhancedRetriever:
    """
    Implémentation spécifique pour LangChain
    """
    def __init__(self, 
                 vector_store,
                 matrix_processor: MatrixProcessor,
                 fallback_enabled: bool = True,
                 metadata_field: str = "idG",
                 search_type: str = "similarity"):
        """
        Initialise le retriever amélioré pour LangChain
        
        Args:
            vector_store: Le vector store LangChain (Chroma, FAISS, etc.)
            matrix_processor: Instance de MatrixProcessor
            fallback_enabled: Si True, utilise les résumés de la Matrice G en fallback
            metadata_field: Nom du champ de métadonnées contenant l'idG
            search_type: Type de recherche à utiliser (similarity, mmr)
        """
        self.vector_store = vector_store
        self.matrix_processor = matrix_processor
        self.fallback_enabled = fallback_enabled
        self.metadata_field = metadata_field
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
        # 1. Détecter les idG pertinents pour la question
        idg_list = self.matrix_processor.detect_idg_from_query(query)
        logger.info(f"idG détectés pour la requête '{query}': {idg_list}")
        
        # 2. Si des idG ont été détectés, ajouter un filtre sur les métadonnées
        results = []
        
        if idg_list:
            # Construire le filtre pour LangChain
            filter_dict = {self.metadata_field: {"$in": idg_list}}
            
            # Effectuer la recherche avec filtre
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            # Si aucun idG n'est détecté, utiliser la recherche standard
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
        
        # 3. Fallback: si aucun résultat n'est trouvé mais des idG sont détectés
        if not results and idg_list and self.fallback_enabled:
            logger.info("Aucun résultat trouvé, utilisation du fallback avec les résumés de la Matrice G")
            fallback_results = self.matrix_processor.get_summaries_from_matrix_g(idg_list)
            
            # Transformer les résultats de fallback pour qu'ils correspondent au format attendu par LangChain
            from langchain.schema import Document
            
            formatted_fallback = []
            for fb in fallback_results:
                doc = Document(
                    page_content=f"Résumé canonique: {fb['titre']} - {fb['résumé']}",
                    metadata={
                        "idG": fb["idG"],
                        "titre": fb["titre"],
                        "type": fb["type"],
                        "is_fallback": True,
                        "source": "matrix_g_fallback"
                    }
                )
                formatted_fallback.append(doc)
            
            return formatted_fallback
        
        return results

# Fonction utilitaire pour créer facilement un retriever amélioré
def create_enhanced_retriever(retriever_type: str, 
                             vector_store: Any,
                             matrix_processor: MatrixProcessor,
                             **kwargs) -> Any:
    """
    Crée un retriever amélioré du type spécifié
    
    Args:
        retriever_type: Type de retriever à créer ("default", "langchain", etc.)
        vector_store: Base de données vectorielle à utiliser
        matrix_processor: Instance de MatrixProcessor
        **kwargs: Arguments supplémentaires pour le retriever
        
    Returns:
        Instance du retriever amélioré
    """
    if retriever_type.lower() == "langchain":
        return LangChainEnhancedRetriever(
            vector_store=vector_store,
            matrix_processor=matrix_processor,
            **kwargs
        )
    else:
        # Retriever par défaut: wrapper générique
        return EnhancedRetrieverWrapper(
            base_retriever=vector_store,
            matrix_processor=matrix_processor,
            **kwargs
        ) 