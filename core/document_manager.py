import os
import logging
from typing import List

from document_processing.vectordb_interface import VectorDBInterface
from document_processing.text_vectorizer import TextVectorizer
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def document_exists(filepath: str) -> bool:
    """
    Vérifie si un document existe déjà dans la base de données.
    
    Args:
        filepath: Chemin du fichier à vérifier
        
    Returns:
        bool: True si le document existe déjà, False sinon
    """
    try:
        # Obtenir le chemin absolu et le nom de fichier
        filepath_abs = os.path.abspath(filepath)
        filename = os.path.basename(filepath_abs)
        
        logger.info(f"Vérification du document: {filepath}")
        logger.info(f"Nom de fichier extrait: {filename}")
        
        # Vérifier si le fichier existe sur le disque
        if not os.path.isfile(filepath):
            logger.warning(f"Le fichier {filepath} n'existe pas sur le disque.")
            # Si c'est un flag de commande, on continue quand même pour vérifier la base de données
            if filepath.startswith("--"):
                logger.debug("Paramètre de ligne de commande détecté, vérification dans la base de données")
            else:
                return False
        
        # Initialiser les gestionnaires
        logger.debug("Initialisation des gestionnaires pour vérification d'existence")
        embeddings_manager = TextVectorizer()
        chroma_manager = VectorDBInterface(embeddings_manager)
        
        # Générer une liste de noms de fichiers possibles (original et orienté)
        # Ceci est nécessaire car le processus peut avoir créé une version "_oriented" du document
        possible_filenames = [filename]
        
        # Vérifier si le fichier est un PDF
        if filename.lower().endswith(".pdf"):
            # Créer le nom de fichier pour la version orientée
            name_without_ext = os.path.splitext(filename)[0]
            oriented_filename = f"{name_without_ext}_oriented.pdf"
            possible_filenames.append(oriented_filename)
            logger.debug(
                f"Ajout du nom de fichier orienté pour vérification: {oriented_filename}"
            )
        
        # Vérifier si des documents avec ce nom existent déjà
        for name in possible_filenames:
            logger.debug(f"Vérification pour le nom de fichier: {name}")
            exists = chroma_manager.document_exists(name)
            if exists:
                logger.info(
                    f"Résultat de la vérification: document existe avec le nom {name}"
                )
                return True
        
        logger.info(
            f"Résultat de la vérification: document {filename} n'existe pas dans la base de données"
        )
        return False
    except Exception as e:
        logger.error(
            f"Erreur lors de la vérification de l'existence du document {filepath}: {e}"
        )
        # Par précaution, retourner False en cas d'erreur
        return False


def delete_document(filepath: str) -> bool:
    """
    Supprime un document de la base de données.
    
    Args:
        filepath: Chemin du fichier à supprimer
        
    Returns:
        bool: True si la suppression a réussi, False sinon
    """
    try:
        # Obtenir le chemin absolu et le nom de fichier
        filepath_abs = os.path.abspath(filepath)
        filename = os.path.basename(filepath_abs)
        
        logger.info(f"Suppression du document: {filepath}")
        logger.info(f"Nom de fichier extrait: {filename}")
        
        # Initialiser les gestionnaires
        embeddings_manager = TextVectorizer()
        chroma_manager = VectorDBInterface(embeddings_manager)
        
        # Générer une liste de noms de fichiers possibles (original et orienté)
        possible_filenames = [filename]
        
        # Vérifier si le fichier est un PDF
        if filename.lower().endswith(".pdf"):
            # Créer le nom de fichier pour la version orientée
            name_without_ext = os.path.splitext(filename)[0]
            oriented_filename = f"{name_without_ext}_oriented.pdf"
            possible_filenames.append(oriented_filename)
            logger.debug(
                f"Ajout du nom de fichier orienté pour suppression: {oriented_filename}"
            )
        
        # Supprimer toutes les versions du document
        success = False
        for name in possible_filenames:
            logger.debug(f"Tentative de suppression pour: {name}")
            result = chroma_manager.delete_document(name)
            if result:
                logger.info(f"Document {name} supprimé avec succès")
                success = True
                
        return success
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du document {filepath}: {e}")
        return False


def cleanup_flag_documents() -> None:
    """
    Nettoie la base de données en supprimant tout document dont le nom commence par '--'.
    Ces documents ont été créés par erreur lorsque des flags de ligne de commande ont été
    traités comme des documents.
    """
    try:
        logger.info(
            "🧹 Vérification des entrées incorrectes dans la base de données..."
        )
        
        # Initialiser les gestionnaires
        embeddings_manager = TextVectorizer()
        chroma_manager = VectorDBInterface(embeddings_manager)
        
        # Obtenir tous les documents de la base de données
        all_documents = chroma_manager.get_all_documents()
        
        # Filtrer pour trouver ceux qui commencent par '--'
        flags_to_delete = []
        for doc_id, metadata in all_documents.items():
            document_name = metadata.get("document_title", "")
            if document_name.startswith("--"):
                flags_to_delete.append(document_name)
        
        # Supprimer les documents de type flag
        if flags_to_delete:
            logger.warning(
                f"⚠️ Détection de {len(flags_to_delete)} drapeaux de commande stockés comme documents."
            )
            for flag in flags_to_delete:
                logger.info(f"🗑️ Suppression de l'entrée incorrecte: {flag}")
                delete_document(flag)
            logger.info("✅ Nettoyage terminé")
        else:
            logger.debug("✅ Aucune entrée incorrecte trouvée dans la base de données")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des flags: {e}")


def get_existing_documents(filepaths: List[str], force_reprocess: bool = False) -> List[str]:
    """
    Identifie les documents existants dans une liste de chemins de fichiers
    
    Args:
        filepaths: Liste des chemins de fichiers à vérifier
        force_reprocess: Si True, retourne une liste vide (ignore les documents existants)
        
    Returns:
        Liste des documents qui existent déjà dans la base de données
    """
    if force_reprocess:
        return []
        
    existing_docs = []
    for filepath in filepaths:
        # Skip any flags (starting with --)
        if filepath.startswith("--"):
            continue
            
        if document_exists(filepath):
            existing_docs.append(filepath)
            
    return existing_docs 