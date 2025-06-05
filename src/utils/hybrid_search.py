from typing import List, Dict, Any, Tuple
from utils.bm25 import BM25
from utils.logger import setup_logger
import numpy as np

# Configurer le logger pour ce module
logger = setup_logger(__file__)

class HybridSearch:
    """
    Classe pour combiner la recherche vectorielle (sémantique) avec BM25 (lexicale)
    
    Paramètres:
        alpha: Poids pour les scores de recherche vectorielle (entre 0 et 1)
        beta: Poids pour les scores BM25 (1 - alpha)
    """
    
    def __init__(self, vector_search, alpha: float = 0.7):
        """
        Initialise la recherche hybride
        
        Args:
            vector_search: Instance du gestionnaire de recherche vectorielle
            alpha: Poids pour les scores vectoriels (1-alpha sera le poids pour BM25)
        """
        self.vector_search = vector_search
        self.bm25 = BM25()
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.documents = []
        self.id_to_doc = {}
        self.is_fitted = False
        
    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """
        Prépare les deux systèmes de recherche
        
        Args:
            documents: Liste de documents à indexer
        """
        # Stocker les documents pour référence future
        self.documents = documents
        
        # Créer un mapping id -> document pour récupérer rapidement les documents
        self.id_to_doc = {doc.get('id', str(i)): doc for i, doc in enumerate(documents)}
        
        # Entraîner BM25 sur les documents
        logger.info("Entraînement de l'index BM25...")
        self.bm25.fit(documents)
        
        # Le modèle vectoriel est supposé déjà entraîné dans le VectorDBInterface
        
        self.is_fitted = True
        logger.info("Indexation hybride terminée")
    
    def search(self, query: str, n_results: int = 5, min_semantic_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride combinant scores vectoriels et BM25
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats à retourner
            min_semantic_score: Score minimal pour les résultats sémantiques
            
        Returns:
            Liste des documents les plus pertinents avec leurs scores
        """
        if not self.is_fitted:
            logger.warning("Le modèle hybride n'a pas été entraîné. Utilisation uniquement de la recherche vectorielle.")
            return self.vector_search.search(query, n_results=n_results)
        
        # Recherche vectorielle
        logger.info("Exécution de la recherche vectorielle...")
        vector_results = self.vector_search.search(query, n_results=n_results * 2)  # Demander plus de résultats pour la fusion
        
        # Extraire les IDs des documents vectoriels et normaliser les scores
        doc_ids = [result.get('id', '') for result in vector_results]
        
        # Convertir les distances en scores (plus la distance est petite, plus le score est élevé)
        vector_scores = [1.0 - result.get('distance', 0.0) for result in vector_results]
        
        # Normaliser les scores vectoriels entre 0 et 1
        if vector_scores:
            max_score = max(vector_scores)
            min_score = min(vector_scores)
            score_range = max_score - min_score
            if score_range > 0:
                vector_scores = [(s - min_score) / score_range for s in vector_scores]
            else:
                vector_scores = [1.0 for _ in vector_scores]
        
        # Créer un dictionnaire ID -> score vectoriel
        vector_id_to_score = {doc_id: score for doc_id, score in zip(doc_ids, vector_scores)}
        
        # Recherche BM25
        logger.info("Exécution de la recherche BM25...")
        bm25_ids, bm25_scores = self.bm25.search(query, topk=n_results * 2)
        
        # Normaliser les scores BM25 entre 0 et 1
        if bm25_scores:
            max_score = max(bm25_scores)
            min_score = min(bm25_scores)
            score_range = max_score - min_score
            if score_range > 0:
                bm25_scores = [(s - min_score) / score_range for s in bm25_scores]
            else:
                bm25_scores = [1.0 for _ in bm25_scores]
        
        # Créer un dictionnaire ID -> score BM25
        bm25_id_to_score = {doc_id: score for doc_id, score in zip(bm25_ids, bm25_scores)}
        
        # Fusion des résultats
        logger.info("Fusion des résultats...")
        
        # Collecter tous les IDs uniques des deux recherches
        all_ids = set(doc_ids) | set(bm25_ids)
        
        # Calculer les scores combinés
        combined_scores = []
        for doc_id in all_ids:
            # Récupérer les scores ou utiliser 0 si non présent
            vector_score = vector_id_to_score.get(doc_id, 0.0)
            bm25_score = bm25_id_to_score.get(doc_id, 0.0)
            
            # Ne considérer que les documents qui ont un score vectoriel minimum
            if vector_score >= min_semantic_score:
                # Calculer le score combiné
                combined_score = self.alpha * vector_score + self.beta * bm25_score
                combined_scores.append((doc_id, combined_score, vector_score, bm25_score))
        
        # Trier par score combiné décroissant
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prendre les n_results meilleurs
        top_combined = combined_scores[:n_results]
        
        # Préparer les résultats finaux
        results = []
        for doc_id, combined_score, vector_score, bm25_score in top_combined:
            if doc_id in self.id_to_doc:
                # Récupérer le document original
                doc = self.id_to_doc[doc_id]
                
                # Ajouter les scores à la sortie
                doc_with_scores = doc.copy()
                doc_with_scores['distance'] = 1.0 - vector_score  # Reconvertir en distance pour compatibilité
                doc_with_scores['bm25_score'] = bm25_score
                doc_with_scores['combined_score'] = combined_score
                doc_with_scores['hybrid_search'] = True
                
                results.append(doc_with_scores)
            else:
                # Si le document n'est pas dans notre cache, le récupérer via la recherche vectorielle
                for vector_doc in vector_results:
                    if vector_doc.get('id', '') == doc_id:
                        vector_doc['bm25_score'] = bm25_score
                        vector_doc['combined_score'] = combined_score
                        vector_doc['hybrid_search'] = True
                        results.append(vector_doc)
                        break
        
        logger.info(f"Recherche hybride terminée: {len(results)} résultats trouvés")
        return results 