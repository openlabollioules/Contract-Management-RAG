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
    
    def __init__(self, vector_search, alpha: float = 0.75):
        """
        Initialise la recherche hybride avec un meilleur équilibre entre vectoriel et lexical
        """
        self.vector_search = vector_search
        # Utiliser les paramètres BM25 optimisés
        self.bm25 = BM25(k1=2.5, b=0.5)
        # Baisse légère d'alpha (de 0.85 à 0.75) pour donner plus de poids aux résultats BM25
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.documents = []
        self.id_to_doc = {}
        self.is_fitted = False
        # Augmentation du nombre minimum de résultats vectoriels à inclure
        self.min_vector_results = 15
        
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
        Recherche hybride avec plus de résultats et moins de filtrage
        """
        if not self.is_fitted:
            logger.warning("Le modèle hybride n'a pas été entraîné. Utilisation uniquement de la recherche vectorielle.")
            return self.vector_search.search(query, n_results=n_results)
        
        # Demander beaucoup plus de résultats pour la recherche vectorielle
        expanded_n = max(n_results * 4, 25)  # Augmenté de 3x à 4x ou 25 minimum
        
        # Recherche vectorielle
        logger.info("Exécution de la recherche vectorielle...")
        vector_results = self.vector_search.search(query, n_results=expanded_n)
        
        # Extraire les IDs et distances
        doc_ids = [result.get('id', '') for result in vector_results]
        distances = [result.get('distance', 1.0) for result in vector_results]
        
        # Convertir les distances en scores
        vector_scores = [1.0 - dist for dist in distances]
        
        # Normaliser les scores vectoriels entre 0 et 1
        if vector_scores:
            max_score = max(vector_scores)
            min_score = min(vector_scores)
            score_range = max_score - min_score
            if score_range > 0:
                vector_scores = [(s - min_score) / score_range for s in vector_scores]
            else:
                vector_scores = [1.0 for _ in vector_scores]
        
        # Analyse de la qualité des résultats vectoriels - adaptation plus fine
        if vector_scores and len(vector_scores) >= 5:
            top_scores = vector_scores[:5]
            avg_top_score = sum(top_scores) / len(top_scores)
            
            # Ajustement plus modéré pour éviter d'exclure des résultats BM25 pertinents
            if avg_top_score > 0.9:  # Seuil augmenté pour être plus sélectif
                adaptive_alpha = min(0.9, self.alpha + 0.1)
                logger.info(f"Résultats vectoriels excellents, alpha ajusté à {adaptive_alpha}")
            elif avg_top_score < 0.4:  # Seuil réduit pour mieux détecter les mauvais résultats
                adaptive_alpha = max(0.6, self.alpha - 0.15)
                logger.info(f"Résultats vectoriels faibles, alpha ajusté à {adaptive_alpha}")
            else:
                adaptive_alpha = self.alpha
        else:
            adaptive_alpha = self.alpha
        
        adaptive_beta = 1.0 - adaptive_alpha
        
        # Créer un dictionnaire ID -> score vectoriel
        vector_id_to_score = {doc_id: score for doc_id, score in zip(doc_ids, vector_scores)}
        
        # Recherche BM25 avec nombre étendu de résultats
        logger.info("Exécution de la recherche BM25...")
        bm25_ids, bm25_scores = self.bm25.search(query, topk=expanded_n * 2)  # Doublé pour trouver plus de résultats
        
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
        logger.info(f"Fusion des résultats avec alpha={adaptive_alpha}, beta={adaptive_beta}...")
        
        # Assurer que tous les meilleurs résultats vectoriels sont inclus
        top_vector_ids = doc_ids[:self.min_vector_results]
        
        # Collecter tous les IDs uniques pour la fusion
        all_ids = set(doc_ids) | set(bm25_ids)
        
        # Calculer les scores combinés avec des ajustements supplémentaires
        combined_scores = []
        for doc_id in all_ids:
            # Récupérer les scores ou utiliser 0 si non présent
            vector_score = vector_id_to_score.get(doc_id, 0.0)
            bm25_score = bm25_id_to_score.get(doc_id, 0.0)
            
            # Pour les meilleurs résultats BM25, ajouter un boost également
            is_top_vector = doc_id in top_vector_ids
            is_top_bm25 = doc_id in bm25_ids[:10]  # Les 10 meilleurs résultats BM25
            
            # Boost pour les résultats qui sont bons dans les deux systèmes
            if is_top_vector and is_top_bm25:
                boost_factor = 1.5  # 50% de boost pour les résultats communs
            elif is_top_vector:
                boost_factor = 1.3  # 30% pour les bons résultats vectoriels
            elif is_top_bm25:
                boost_factor = 1.2  # 20% pour les bons résultats BM25
            else:
                boost_factor = 1.0
            
            # Calculer le score combiné avec boost
            combined_score = (adaptive_alpha * vector_score + adaptive_beta * bm25_score) * boost_factor
            combined_scores.append((doc_id, combined_score, vector_score, bm25_score))
        
        # Trier par score combiné décroissant
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prendre les n_results meilleurs
        top_combined = combined_scores[:n_results]
        
        # S'assurer que les top_min_vector_results vectoriels sont inclus
        included_vector_ids = {doc_id for doc_id, _, _, _ in top_combined if doc_id in top_vector_ids}
        missing_vector_count = min(self.min_vector_results, len(top_vector_ids)) - len(included_vector_ids)
        
        if missing_vector_count > 0:
            # Identifier les IDs vectoriels manquants
            missing_vector_ids = [doc_id for doc_id in top_vector_ids 
                                if doc_id not in included_vector_ids][:missing_vector_count]
            
            # Ajouter ces résultats manquants
            for doc_id in missing_vector_ids:
                vector_score = vector_id_to_score.get(doc_id, 0.0)
                bm25_score = bm25_id_to_score.get(doc_id, 0.0)
                combined_score = adaptive_alpha * vector_score + adaptive_beta * bm25_score
                
                # Remplacer le résultat le moins bien classé
                if len(top_combined) >= n_results:
                    top_combined.pop()  # Retirer le dernier élément
                
                # Ajouter le résultat vectoriel et retrier
                top_combined.append((doc_id, combined_score, vector_score, bm25_score))
                top_combined.sort(key=lambda x: x[1], reverse=True)
        
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
                doc_with_scores['search_type'] = 'vector' if doc_id in top_vector_ids else 'hybrid'
                
                results.append(doc_with_scores)
            else:
                # Si le document n'est pas dans notre cache, le récupérer via la recherche vectorielle
                for vector_doc in vector_results:
                    if vector_doc.get('id', '') == doc_id:
                        vector_doc['bm25_score'] = bm25_score
                        vector_doc['combined_score'] = combined_score
                        vector_doc['hybrid_search'] = True
                        vector_doc['search_type'] = 'vector' if doc_id in top_vector_ids else 'hybrid'
                        results.append(vector_doc)
                        break
        
        logger.info(f"Recherche hybride terminée: {len(results)} résultats trouvés")
        return results 