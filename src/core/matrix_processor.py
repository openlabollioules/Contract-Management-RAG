import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from loguru import logger

# Chemins de données
DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))
MATRIX_G_PATH = DATA_DIR / "matrix_g.csv"  # À adapter selon votre structure
MATRIX_DOC_PATH = DATA_DIR / "matrix_doc.csv"  # À adapter selon votre structure

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75

class MatrixProcessor:
    def __init__(self, 
                 matrix_g_path: Union[str, Path] = MATRIX_G_PATH,
                 matrix_doc_path: Union[str, Path] = MATRIX_DOC_PATH,
                 embedding_model: str = EMBEDDING_MODEL,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialise le processeur de matrices pour le pipeline RAG de contrats.
        
        Args:
            matrix_g_path: Chemin vers le fichier de la Matrice G (thésaurus de clauses)
            matrix_doc_path: Chemin vers le fichier de la Matrice Doc (clauses de contrats)
            embedding_model: Modèle d'embedding à utiliser
            similarity_threshold: Seuil de similarité pour la classification
        """
        # Paramètres
        self.similarity_threshold = similarity_threshold
        
        # Chargement des matrices
        logger.info(f"Chargement de la Matrice G depuis {matrix_g_path}")
        self.matrix_g = self._load_matrix_g(matrix_g_path)
        
        logger.info(f"Chargement de la Matrice Doc depuis {matrix_doc_path}")
        self.matrix_doc = self._load_matrix_doc(matrix_doc_path)
        
        # Chargement du modèle d'embedding
        logger.info(f"Chargement du modèle d'embedding {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Construction de l'index FAISS pour les résumés/titres de la Matrice G
        logger.info("Construction de l'index FAISS pour la Matrice G")
        self.faiss_index, self.index_id_mapping = self._build_faiss_index()
        
        # Extraction des tags et mots-clés de la Matrice G pour la détection rapide
        self.idg_tag_mapping = self._build_tag_mapping()
        
        logger.info("Initialisation de MatrixProcessor terminée")
    
    def _load_matrix_g(self, path: Union[str, Path]) -> pd.DataFrame:
        """Charge la Matrice G depuis un fichier CSV ou Excel"""
        if str(path).endswith('.csv'):
            df = pd.read_csv(path)
        elif str(path).endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Format de fichier non supporté pour la Matrice G: {path}")
        
        # Vérification des colonnes requises
        required_cols = ["idG", "titre", "type", "tags", "résumé"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne requise manquante dans la Matrice G: {col}")
        
        return df
    
    def _load_matrix_doc(self, path: Union[str, Path]) -> pd.DataFrame:
        """Charge la Matrice Doc depuis un fichier CSV ou Excel"""
        if str(path).endswith('.csv'):
            df = pd.read_csv(path)
        elif str(path).endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Format de fichier non supporté pour la Matrice Doc: {path}")
        
        # Vérification des colonnes requises
        required_cols = ["idDoc", "texte", "idG"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne requise manquante dans la Matrice Doc: {col}")
        
        return df
    
    def _build_faiss_index(self) -> Tuple[faiss.IndexFlatIP, Dict[int, str]]:
        """
        Construit un index FAISS pour les résumés et titres de la Matrice G
        
        Returns:
            Tuple contenant (index FAISS, mapping des indices aux idG)
        """
        # Combiner titre et résumé pour une meilleure représentation
        texts = []
        for _, row in self.matrix_g.iterrows():
            combined_text = f"{row['titre']} - {row['type']} - {row['résumé']}"
            texts.append(combined_text)
        
        # Générer les embeddings
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Créer l'index FAISS (inner product pour la similarité cosinus)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # Créer le mapping des indices vers les idG
        index_id_mapping = {i: str(self.matrix_g.iloc[i]['idG']) for i in range(len(self.matrix_g))}
        
        return index, index_id_mapping
    
    def _build_tag_mapping(self) -> Dict[str, List[str]]:
        """
        Construit un mapping entre les tags et les idG correspondants
        
        Returns:
            Dictionnaire avec les tags comme clés et les listes d'idG comme valeurs
        """
        tag_to_idg = {}
        
        for _, row in self.matrix_g.iterrows():
            idg = str(row['idG'])
            
            # Extraire les tags et les traiter
            tags = row['tags']
            if isinstance(tags, str):
                # Séparation des tags (on suppose qu'ils sont séparés par des virgules ou des points-virgules)
                tag_list = [tag.strip().lower() for tag in re.split(r'[,;]', tags)]
                
                for tag in tag_list:
                    if tag not in tag_to_idg:
                        tag_to_idg[tag] = []
                    tag_to_idg[tag].append(idg)
        
        return tag_to_idg
    
    def classify_clause(self, clause_text: str) -> Optional[str]:
        """
        Classifie une clause de contrat pour trouver l'idG correspondant le plus pertinent
        
        Args:
            clause_text: Texte de la clause à classifier
            
        Returns:
            idG le plus pertinent ou None si aucune correspondance ne dépasse le seuil
        """
        # Générer l'embedding de la clause
        clause_embedding = self.embedding_model.encode([clause_text], normalize_embeddings=True)
        
        # Rechercher dans l'index FAISS
        scores, indices = self.faiss_index.search(np.array(clause_embedding).astype('float32'), k=1)
        
        # Vérifier si la similarité dépasse le seuil
        if scores[0][0] >= self.similarity_threshold:
            # Retourner l'idG correspondant
            return self.index_id_mapping[indices[0][0]]
        else:
            return None
    
    def detect_idg_from_query(self, question_text: str) -> List[str]:
        """
        Détecte les idG pertinents à partir d'une question utilisateur
        
        Args:
            question_text: Texte de la question utilisateur
            
        Returns:
            Liste des idG pertinents
        """
        # Approche 1: Recherche par mot-clé/tag
        detected_idgs = set()
        
        # Convertir la question en minuscules pour la correspondance
        question_lower = question_text.lower()
        
        # Vérifier si des tags connus sont présents dans la question
        for tag, idg_list in self.idg_tag_mapping.items():
            if tag in question_lower:
                for idg in idg_list:
                    detected_idgs.add(idg)
        
        # Approche 2: Recherche vectorielle dans les résumés/titres
        # Générer l'embedding de la question
        query_embedding = self.embedding_model.encode([question_text], normalize_embeddings=True)
        
        # Rechercher dans l'index FAISS les 3 plus proches
        scores, indices = self.faiss_index.search(np.array(query_embedding).astype('float32'), k=3)
        
        # Ajouter les idG correspondants si la similarité est suffisante
        for i in range(len(indices[0])):
            if scores[0][i] >= self.similarity_threshold * 0.8:  # Seuil légèrement réduit pour la recherche
                detected_idgs.add(self.index_id_mapping[indices[0][i]])
        
        return list(detected_idgs)
    
    def get_clause_by_idg(self, idg_list: List[str]) -> List[Dict]:
        """
        Récupère les clauses correspondant aux idG spécifiés
        
        Args:
            idg_list: Liste des idG à rechercher
            
        Returns:
            Liste des clauses correspondantes avec leurs métadonnées
        """
        results = []
        
        # Filtrer la Matrice Doc pour ne conserver que les clauses avec les idG spécifiés
        filtered_df = self.matrix_doc[self.matrix_doc['idG'].astype(str).isin(idg_list)]
        
        for _, row in filtered_df.iterrows():
            clause = {
                'idDoc': row['idDoc'],
                'texte': row['texte'],
                'idG': row['idG']
            }
            
            # Ajouter les métadonnées supplémentaires si elles existent
            for col in self.matrix_doc.columns:
                if col not in ['idDoc', 'texte', 'idG']:
                    clause[col] = row[col]
            
            results.append(clause)
        
        return results
    
    def get_summaries_from_matrix_g(self, idg_list: List[str]) -> List[Dict]:
        """
        Récupère les résumés de la Matrice G pour les idG spécifiés (fallback)
        
        Args:
            idg_list: Liste des idG à rechercher
            
        Returns:
            Liste des résumés correspondants
        """
        results = []
        
        # Filtrer la Matrice G pour ne conserver que les entrées avec les idG spécifiés
        filtered_df = self.matrix_g[self.matrix_g['idG'].astype(str).isin(idg_list)]
        
        for _, row in filtered_df.iterrows():
            entry = {
                'idG': row['idG'],
                'titre': row['titre'],
                'type': row['type'],
                'résumé': row['résumé'],
                'is_fallback': True  # Marquer comme fallback
            }
            
            results.append(entry)
        
        return results

# Module de recherche amélioré avec filtrage par idG
class EnhancedRetriever:
    def __init__(self, vector_db, matrix_processor: MatrixProcessor):
        """
        Initialise le module de recherche amélioré
        
        Args:
            vector_db: Base de données vectorielle (ChromaDB, etc.)
            matrix_processor: Instance de MatrixProcessor
        """
        self.vector_db = vector_db
        self.matrix_processor = matrix_processor
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Exécute une requête avec filtrage par idG
        
        Args:
            query_text: Texte de la requête utilisateur
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des résultats pertinents
        """
        # 1. Détecter les idG pertinents pour la question
        idg_list = self.matrix_processor.detect_idg_from_query(query_text)
        logger.info(f"idG détectés pour la requête: {idg_list}")
        
        # 2. Si des idG ont été détectés, filtrer la recherche vectorielle
        results = []
        if idg_list:
            # Filtrer les documents qui ont les idG correspondants
            # Note: Adapter cette partie selon votre implémentation de vector_db
            metadata_filter = {"idG": {"$in": idg_list}}
            results = self.vector_db.query(
                query_text=query_text,
                n_results=top_k,
                where=metadata_filter
            )
        else:
            # Recherche sans filtre si aucun idG n'est détecté
            results = self.vector_db.query(
                query_text=query_text,
                n_results=top_k
            )
        
        # 3. Fallback: si aucun résultat n'est trouvé mais que des idG ont été détectés
        if not results and idg_list:
            logger.info("Aucun résultat trouvé, utilisation du fallback avec les résumés de la Matrice G")
            fallback_results = self.matrix_processor.get_summaries_from_matrix_g(idg_list)
            return fallback_results
        
        return results
