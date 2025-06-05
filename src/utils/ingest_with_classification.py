#!/usr/bin/env python3
"""
ingest_with_classification.py - Extension du pipeline d'ingestion pour classifier les clauses
"""

import os
import sys
import argparse
from pathlib import Path
import json
import chromadb
from tqdm import tqdm
from loguru import logger

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer les modules nécessaires
from src.core.matrix_processor import MatrixProcessor
from src.utils.logger import init_logger

def parse_arguments():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Ingestion de documents avec classification des clauses")
    
    parser.add_argument(
        "--matrix-g", 
        required=True,
        help="Chemin vers le fichier de la Matrice G"
    )
    
    parser.add_argument(
        "--input-chunks",
        required=True,
        help="Fichier JSON contenant les chunks à traiter"
    )
    
    parser.add_argument(
        "--output-chunks",
        required=True,
        help="Fichier JSON de sortie avec les chunks classifiés"
    )
    
    parser.add_argument(
        "--chroma-path", 
        default="./chroma_db",
        help="Chemin vers la base de données Chroma"
    )
    
    parser.add_argument(
        "--collection-name",
        default="contract_chunks",
        help="Nom de la collection Chroma à utiliser"
    )
    
    parser.add_argument(
        "--index-chroma",
        action="store_true",
        help="Indexer les chunks dans ChromaDB en plus de les classifier"
    )
    
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=50,
        help="Nombre minimum de tokens pour qu'un chunk soit considéré comme une clause"
    )
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Nombre maximum de chunks à traiter (0 = tous)"
    )
    
    return parser.parse_args()

def load_chunks(input_file):
    """Charge les chunks depuis un fichier JSON"""
    logger.info(f"Chargement des chunks depuis {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Chargé {len(chunks)} chunks")
    return chunks

def classify_chunks(chunks, matrix_processor, min_tokens=50, max_chunks=0):
    """
    Classifie les chunks en utilisant le MatrixProcessor
    
    Args:
        chunks: Liste des chunks à classifier
        matrix_processor: Instance de MatrixProcessor
        min_tokens: Nombre minimum de tokens pour qu'un chunk soit considéré comme une clause
        max_chunks: Nombre maximum de chunks à traiter (0 = tous)
        
    Returns:
        Liste des chunks avec les classifications
    """
    logger.info("Classification des chunks...")
    
    # Limiter le nombre de chunks si demandé
    if max_chunks > 0 and max_chunks < len(chunks):
        chunks_to_process = chunks[:max_chunks]
        logger.info(f"Traitement limité à {max_chunks} chunks")
    else:
        chunks_to_process = chunks
    
    # Classifier chaque chunk
    for chunk in tqdm(chunks_to_process, desc="Classification des chunks"):
        # Vérifier si le chunk est suffisamment long pour être une clause
        text = chunk.get("text", "")
        token_count = len(text.split())
        
        if token_count >= min_tokens:
            # Classifier le chunk
            idg = matrix_processor.classify_clause(text)
            
            # Ajouter la classification aux métadonnées
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            
            if idg:
                chunk["metadata"]["idG"] = idg
                
                # Récupérer les informations supplémentaires de la Matrice G
                idg_info = matrix_processor.get_summaries_from_matrix_g([idg])
                if idg_info:
                    info = idg_info[0]
                    chunk["metadata"]["clause_titre"] = info.get("titre", "")
                    chunk["metadata"]["clause_type"] = info.get("type", "")
                    
                    # Récupérer les tags si disponibles
                    tags_row = matrix_processor.matrix_g[matrix_processor.matrix_g['idG'].astype(str) == idg]
                    if not tags_row.empty and 'tags' in tags_row.columns:
                        tags = tags_row.iloc[0]['tags']
                        if tags and isinstance(tags, str):
                            chunk["metadata"]["clause_tags"] = tags
    
    logger.info(f"Classification terminée. {sum(1 for c in chunks_to_process if 'metadata' in c and 'idG' in c['metadata'])} chunks classifiés")
    return chunks

def index_chunks_in_chroma(chunks, chroma_client, collection_name):
    """
    Indexe les chunks dans ChromaDB
    
    Args:
        chunks: Liste des chunks à indexer
        chroma_client: Client ChromaDB
        collection_name: Nom de la collection
        
    Returns:
        Nombre de chunks indexés
    """
    logger.info(f"Indexation des chunks dans ChromaDB (collection: {collection_name})...")
    
    # Créer ou récupérer la collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Préparer les données pour l'indexation
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        # Vérifier si le chunk a un ID
        if "id" in chunk:
            chunk_id = str(chunk["id"])
        else:
            chunk_id = f"chunk_{i}"
        
        # Vérifier si le chunk a du texte
        text = chunk.get("text", "")
        if not text:
            continue
        
        # Ajouter l'ID
        ids.append(chunk_id)
        
        # Ajouter le document
        documents.append(text)
        
        # Préparer les métadonnées
        metadata = chunk.get("metadata", {})
        
        # S'assurer que toutes les valeurs sont des types JSON valides
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_metadata[key] = value
            else:
                cleaned_metadata[key] = str(value)
        
        metadatas.append(cleaned_metadata)
    
    # Ajouter les documents à la collection par lots
    batch_size = 100
    total_indexed = 0
    
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        batch_ids = ids[i:end]
        batch_documents = documents[i:end]
        batch_metadatas = metadatas[i:end]
        
        # Indexer le lot
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
        total_indexed += len(batch_ids)
        logger.info(f"Indexé {total_indexed}/{len(ids)} chunks")
    
    logger.info(f"Indexation terminée. {total_indexed} chunks indexés dans ChromaDB")
    return total_indexed

def save_chunks(chunks, output_file):
    """Sauvegarde les chunks dans un fichier JSON"""
    logger.info(f"Sauvegarde des chunks dans {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sauvegardé {len(chunks)} chunks")

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
        matrix_doc_path=None  # Nous n'avons pas besoin de la Matrice Doc ici
    )
    
    # Charger les chunks
    chunks = load_chunks(args.input_chunks)
    
    # Classifier les chunks
    classified_chunks = classify_chunks(
        chunks=chunks,
        matrix_processor=matrix_processor,
        min_tokens=args.min_tokens,
        max_chunks=args.max_chunks
    )
    
    # Sauvegarder les chunks classifiés
    save_chunks(classified_chunks, args.output_chunks)
    
    # Indexer les chunks dans ChromaDB si demandé
    if args.index_chroma:
        # Initialiser le client ChromaDB
        logger.info(f"Connexion à ChromaDB à {args.chroma_path}")
        chroma_client = chromadb.PersistentClient(path=args.chroma_path)
        
        # Indexer les chunks
        index_chunks_in_chroma(
            chunks=classified_chunks,
            chroma_client=chroma_client,
            collection_name=args.collection_name
        )
    
    logger.info("Traitement terminé avec succès")

if __name__ == "__main__":
    main() 