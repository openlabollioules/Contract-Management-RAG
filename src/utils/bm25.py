import numpy as np
import re
from typing import List, Dict, Tuple, Any
from collections import Counter

class BM25:
    """
    Implémentation de l'algorithme BM25 pour la recherche lexicale
    
    Paramètres:
        k1: Paramètre de saturation de la fréquence des termes (1.2-2.0 typiquement)
        b: Paramètre de normalisation de la longueur des documents (0.75 typiquement)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenized_docs = []
        self.doc_ids = []
        
    def preprocess(self, text: str) -> List[str]:
        """
        Prétraitement simple du texte: mise en minuscule, 
        suppression des caractères spéciaux et tokenisation
        """
        text = text.lower()
        # Suppression des caractères spéciaux tout en conservant les caractères accentués
        text = re.sub(r'[^\w\s\-àáâãäåçèéêëìíîïñòóôõöùúûüýÿ]', ' ', text)
        # Remplacement des séquences d'espaces par un seul espace
        text = re.sub(r'\s+', ' ', text)
        # Tokenisation simple par espace
        return text.strip().split()
    
    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """
        Construit l'index BM25 à partir d'une liste de documents
        
        Args:
            documents: Liste de dictionnaires contenant les documents et leurs métadonnées
        """
        self.doc_ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        
        # Tokenize documents
        self.tokenized_docs = []
        for doc in documents:
            content = doc.get('document', '')
            self.tokenized_docs.append(self.preprocess(content))
        
        # Calculer les longueurs des documents
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.corpus_size = len(self.tokenized_docs)
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculer la fréquence des documents pour chaque terme
        self.doc_freqs = []
        for i in range(self.corpus_size):
            doc_freq = Counter(self.tokenized_docs[i])
            self.doc_freqs.append(doc_freq)
        
        # Calculer l'IDF pour chaque terme
        self._calc_idf()
    
    def _calc_idf(self) -> None:
        """
        Calcule les valeurs IDF pour tous les termes du corpus
        """
        idf_sum = 0
        unique_terms = set()
        for doc in self.tokenized_docs:
            unique_terms.update(doc)
        
        # Calculer l'IDF pour chaque terme
        for term in unique_terms:
            n_docs_with_term = sum(1 for doc in self.tokenized_docs if term in doc)
            idf = np.log((self.corpus_size - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)
            self.idf[term] = idf
            idf_sum += idf
        
        # Normaliser les valeurs IDF
        for term, value in self.idf.items():
            self.idf[term] = value / idf_sum if idf_sum > 0 else 0
    
    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche des documents correspondant à la requête
        
        Args:
            query: Requête de recherche
            topk: Nombre de résultats à retourner
            
        Returns:
            Liste des documents les plus pertinents avec leurs scores
        """
        query_tokens = self.preprocess(query)
        
        scores = []
        for i in range(self.corpus_size):
            score = self._score(query_tokens, i)
            scores.append((self.doc_ids[i], score))
        
        # Trier par score décroissant et récupérer le top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [id for id, _ in scores[:topk]]
        top_scores = [score for _, score in scores[:topk]]
        
        return top_ids, top_scores
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calcule le score BM25 d'un document pour une requête
        
        Args:
            query_tokens: Tokens de la requête
            doc_idx: Index du document
            
        Returns:
            Score BM25 du document
        """
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
                
            freq = doc_freq.get(token, 0)
            numerator = self.idf[token] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator if denominator != 0 else 0
            
        return score 