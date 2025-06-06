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
    
    def __init__(self, k1: float = 2.5, b: float = 0.5):
        # Paramètres optimisés: k1 plus élevé favorise les termes répétés importants,
        # b réduit pour donner moins d'importance à la longueur des documents
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenized_docs = []
        self.doc_ids = []
        # Liste de termes techniques importants à privilégier
        self.technical_terms = [
            'mw', 'kw', 'kwh', 'mwh', 'puissance', 'capacité', 'v2043', '1197',
            'garantie', 'warrant', 'risque', 'change', 'rub', 'eur', 'taux', 
            'clause', 'section', 'article', 'test', 'performance', 'heat', 'balance',
            'technique', 'diagramme', 'annexe', 'contrat', 'partie', 'responsabilité'
        ]
        
    def preprocess(self, text: str) -> List[str]:
        """
        Prétraitement amélioré pour les contrats avec focus sur termes techniques
        """
        text = text.lower()
        
        # Conservation améliorée des chiffres, pourcentages et symboles importants
        text = re.sub(r'[^\w\s\-àáâãäåçèéêëìíîïñòóôõöùúûüýÿ\d%€$\.,:/]', ' ', text)
        
        # Traitement spécial pour les dates (format JJ/MM/AAAA ou AAAA-MM-JJ)
        text = re.sub(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})', r'date_\1_\2_\3', text)
        text = re.sub(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', r'date_\3_\2_\1', text)
        
        # Traitement des références de clauses et articles
        text = re.sub(r'(article|section|clause|paragraphe|annexe|§)\s*(\d+(\.\d+)?)', r'\1_\2', text)
        
        # Traitement amélioré des valeurs numériques avec unités (crucial)
        text = re.sub(r'(\d+(\.\d+)?)\s*(mw|kw|mwh|kwh|kg|m²|m2|mm|cm|m3|%|eur|rub|usd)', r'\1_\3', text)
        
        # Fusion des espaces dans les grands nombres (puissances, montants)
        text = re.sub(r'(\d{1,3})\s+(\d{3})', r'\1\2', text)
        
        # Traitement des codes techniques spécifiques (ex: 75V2043-007d)
        text = re.sub(r'(\d+[a-z][\d-]+[a-z\d]*)', r'code_\1', text)
        
        # Remplacement des séquences d'espaces par un seul espace
        text = re.sub(r'\s+', ' ', text)
        
        tokens = text.strip().split()
        
        # Donner plus de poids aux termes techniques en les dupliquant
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            # Si c'est un terme technique ou contient une unité/valeur, le dupliquer
            if any(term in token for term in self.technical_terms) or re.match(r'\d+_\w+', token):
                expanded_tokens.append(token)  # Duplication pour augmenter le poids
            
        return expanded_tokens
    
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
        Calcule les valeurs IDF avec des ajustements pour les termes techniques
        """
        unique_terms = set()
        for doc in self.tokenized_docs:
            unique_terms.update(doc)
        
        # Calculer l'IDF pour chaque terme
        for term in unique_terms:
            n_docs_with_term = sum(1 for doc in self.tokenized_docs if term in doc)
            idf = np.log((self.corpus_size - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)
            
            # Boost pour les termes techniques et les valeurs numériques
            if any(tech_term in term for tech_term in self.technical_terms):
                idf *= 1.5  # Boost de 50% pour les termes techniques
            elif re.match(r'\d+_\w+', term):  # Valeurs avec unités (1197_mw)
                idf *= 2.0  # Boost de 100% pour les valeurs numériques avec unités
            elif 'date_' in term:  # Dates formatées
                idf *= 1.8  # Boost de 80% pour les dates
            
            self.idf[term] = idf
        
        # Normaliser les valeurs IDF par rapport au maximum (au lieu de la somme)
        max_idf = max(self.idf.values()) if self.idf else 1.0
        for term, value in self.idf.items():
            self.idf[term] = value / max_idf if max_idf > 0 else 0
    
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