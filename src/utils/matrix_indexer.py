#!/usr/bin/env python3
"""
matrix_indexer.py - Outil pour indexer les matrices G et Doc dans ChromaDB
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import chromadb
from loguru import logger

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer les modules nécessaires
from src.core.matrix_processor import MatrixProcessor
from src.utils.logger import init_logger

def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Indexer les matrices G et Doc dans ChromaDB")
    
    parser.add_argument(
        "--matrix-g", 
        required=True,
        help="Chemin vers le fichier de la Matrice G (CSV ou Excel)"
    )
    
    parser.add_argument(
        "--matrix-doc", 
        required=True,
        help="Chemin vers le fichier de la Matrice Doc (CSV ou Excel)"
    )
    
    parser.add_argument(
        "--chroma-path", 
        default="./chroma_db",
        help="Chemin vers la base de données Chroma"
    )
    
    parser.add_argument(
        "--collection-g",
        default="matrix_g",
        help="Nom de la collection pour la Matrice G"
    )
    
    parser.add_argument(
        "--collection-doc",
        default="matrix_doc",
        help="Nom de la collection pour la Matrice Doc"
    )
    
    parser.add_argument(
        "--classify-doc",
        action="store_true",
        help="Classifier automatiquement les clauses de Doc sans idG"
    )
    
    return parser.parse_args()

def index_matrix_g(matrix_processor, chroma_client, collection_name):
    """
    Indexe la Matrice G dans ChromaDB
    
    Args:
        matrix_processor: Instance de MatrixProcessor
        chroma_client: Client ChromaDB
        collection_name: Nom de la collection
    """
    logger.info(f"Indexation de la Matrice G dans la collection {collection_name}")
    
    # Créer ou récupérer la collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Préparer les données pour l'indexation
    ids = []
    documents = []
    metadatas = []
    
    # Parcourir les entrées de la Matrice G
    for _, row in tqdm(matrix_processor.matrix_g.iterrows(), total=len(matrix_processor.matrix_g)):
        # Identifiant unique
        idg = str(row['idG'])
        ids.append(f"idg_{idg}")
        
        # Document (combinaison du titre, type et résumé)
        document = f"{row['titre']} - {row['type']} - {row['résumé']}"
        documents.append(document)
        
        # Métadonnées
        metadata = {
            "idG": idg,
            "titre": row['titre'],
            "type": row['type'],
            "source": "matrix_g"
        }
        
        # Ajouter les tags s'ils existent
        if 'tags' in row and pd.notna(row['tags']):
            metadata["tags"] = row['tags']
        
        metadatas.append(metadata)
    
    # Ajouter les documents à la collection
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Ajout de {len(ids)} entrées de la Matrice G à ChromaDB")
    else:
        logger.warning("Aucune entrée à indexer dans la Matrice G")

def index_matrix_doc(matrix_processor, chroma_client, collection_name, classify=False):
    """
    Indexe la Matrice Doc dans ChromaDB
    
    Args:
        matrix_processor: Instance de MatrixProcessor
        chroma_client: Client ChromaDB
        collection_name: Nom de la collection
        classify: Si True, classifie automatiquement les clauses sans idG
    """
    logger.info(f"Indexation de la Matrice Doc dans la collection {collection_name}")
    
    # Créer ou récupérer la collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Préparer les données pour l'indexation
    ids = []
    documents = []
    metadatas = []
    
    # Parcourir les entrées de la Matrice Doc
    for _, row in tqdm(matrix_processor.matrix_doc.iterrows(), total=len(matrix_processor.matrix_doc)):
        # Identifiant unique
        iddoc = str(row['idDoc'])
        ids.append(f"iddoc_{iddoc}")
        
        # Document (texte de la clause)
        document = row['texte']
        documents.append(document)
        
        # Déterminer l'idG
        idg = None
        if 'idG' in row and pd.notna(row['idG']):
            idg = str(row['idG'])
        elif classify:
            # Classifier la clause si demandé
            idg = matrix_processor.classify_clause(document)
            if idg:
                logger.info(f"Clause {iddoc} classifiée automatiquement comme {idg}")
        
        # Métadonnées
        metadata = {
            "idDoc": iddoc,
            "source": "matrix_doc"
        }
        
        # Ajouter l'idG s'il existe
        if idg:
            metadata["idG"] = idg
        
        # Ajouter les autres métadonnées disponibles
        for col in matrix_processor.matrix_doc.columns:
            if col not in ['idDoc', 'texte', 'idG'] and pd.notna(row[col]):
                metadata[col] = row[col]
        
        metadatas.append(metadata)
    
    # Ajouter les documents à la collection
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Ajout de {len(ids)} entrées de la Matrice Doc à ChromaDB")
    else:
        logger.warning("Aucune entrée à indexer dans la Matrice Doc")

def main():
    """Fonction principale"""
    # Initialiser le logger
    init_logger()
    
    # Parser les arguments
    args = parse_arguments()
    
    # Initialiser le processeur de matrices
    logger.info("Initialisation du processeur de matrices")
    matrix_processor = MatrixProcessor(
        matrix_g_path=args.matrix_g,
        matrix_doc_path=args.matrix_doc
    )
    
    # Initialiser le client ChromaDB
    logger.info(f"Connexion à ChromaDB à {args.chroma_path}")
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    
    # Indexer la Matrice G
    index_matrix_g(
        matrix_processor=matrix_processor,
        chroma_client=chroma_client,
        collection_name=args.collection_g
    )
    
    # Indexer la Matrice Doc
    index_matrix_doc(
        matrix_processor=matrix_processor,
        chroma_client=chroma_client,
        collection_name=args.collection_doc,
        classify=args.classify_doc
    )
    
    logger.info("Indexation terminée avec succès")

if __name__ == "__main__":
    main() 