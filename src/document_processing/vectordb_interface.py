import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils.logger import setup_logger

from .text_vectorizer import TextVectorizer

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class VectorDBInterface:
    def __init__(
        self,
        embeddings_manager: TextVectorizer,
        persist_directory: str = "chroma_db",
        collection_name: str = "contracts",
    ):
        """
        Initialize ChromaDB manager with an embeddings manager

        Args:
            embeddings_manager: Instance of TextVectorizer for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.embeddings_manager = embeddings_manager
        self.persist_directory = persist_directory
        logger.info(
            f"Initialisation de VectorDBInterface (persist_directory={persist_directory}, collection={collection_name})"
        )

        # Initialize ChromaDB client
        logger.debug(f"Création du client ChromaDB persistant")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or get collection
        logger.debug(f"Récupération ou création de la collection '{collection_name}'")
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{collection_name}' initialisée")

    def document_exists(self, filename: str) -> bool:
        """
        Vérifie si un document avec le nom de fichier donné existe déjà dans la base de données

        Args:
            filename: Nom du fichier à vérifier

        Returns:
            bool: True si le document existe, False sinon
        """
        logger.info(f"Vérification de l'existence du document: {filename}")

        try:
            # Récupérer tous les documents et leurs métadonnées pour une vérification complète
            all_docs = self.collection.get()

            if not all_docs or "metadatas" not in all_docs or not all_docs["metadatas"]:
                logger.debug("Aucun document trouvé dans la base de données")
                return False

            # Convertir le nom du fichier en minuscules pour une comparaison insensible à la casse
            filename_lower = filename.lower()

            # Rechercher le nom de fichier dans les métadonnées de document_title ou filename
            doc_exists = False
            for i, metadata in enumerate(all_docs["metadatas"]):
                # Vérifier dans document_title
                if "document_title" in metadata:
                    doc_title = (
                        metadata["document_title"].lower()
                        if metadata["document_title"]
                        else ""
                    )
                    logger.debug(
                        f"Document en base: '{metadata['document_title']}', à comparer avec: '{filename}'"
                    )

                    # Vérifier si le nom du fichier est dans le titre du document
                    if filename_lower in doc_title:
                        logger.info(
                            f"Document trouvé dans la base de données: '{metadata['document_title']}' contient '{filename}'"
                        )
                        doc_exists = True
                        break

                # Vérifier dans filename (si disponible)
                if "filename" in metadata and metadata["filename"]:
                    db_filename = metadata["filename"].lower()
                    logger.debug(
                        f"Vérification avec filename: '{db_filename}' vs '{filename_lower}'"
                    )

                    if filename_lower == db_filename or filename_lower in db_filename:
                        logger.info(
                            f"Document trouvé dans la base de données par filename: '{db_filename}'"
                        )
                        doc_exists = True
                        break

            if not doc_exists:
                # Essayer une méthode alternative de recherche par contenu
                logger.debug(
                    "Tentative de recherche alternative dans le contenu des documents"
                )
                try:
                    # Utiliser la recherche sémantique pour trouver des documents avec le nom du fichier
                    results = self.collection.query(query_texts=[filename], n_results=5)

                    if len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                        # Vérifier si un de ces documents contient vraiment le nom du fichier
                        for i, doc in enumerate(results["documents"][0]):
                            if filename_lower in doc.lower():
                                logger.info(
                                    f"Document trouvé par recherche sémantique qui contient '{filename}'"
                                )
                                doc_exists = True
                                break
                except Exception as e:
                    logger.warning(f"Erreur lors de la recherche alternative: {e}")

            logger.info(
                f"Résultat final de la vérification - Document '{filename}' existe: {doc_exists}"
            )
            return doc_exists

        except Exception as e:
            logger.warning(
                f"Erreur lors de la vérification de l'existence du document: {e}"
            )
            # Par précaution, retourner False en cas d'erreur
            return False

    def delete_document(self, filename: str) -> bool:
        """
        Supprime tous les chunks associés à un document spécifique

        Args:
            filename: Nom du fichier du document à supprimer

        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        logger.warning(f"Suppression du document: {filename}")

        try:
            # Récupérer tous les documents et leurs métadonnées
            all_docs = self.collection.get()

            if not all_docs or "metadatas" not in all_docs or not all_docs["metadatas"]:
                logger.warning(f"Aucun document trouvé pour la suppression")
                return False

            # Convertir le nom du fichier en minuscules pour une comparaison insensible à la casse
            filename_lower = filename.lower()

            # Collecter les IDs des chunks à supprimer
            chunk_ids_to_delete = []

            for i, metadata in enumerate(all_docs["metadatas"]):
                # Vérifier dans document_title
                if "document_title" in metadata and metadata["document_title"]:
                    doc_title = metadata["document_title"].lower()
                    if filename_lower in doc_title:
                        chunk_ids_to_delete.append(all_docs["ids"][i])
                        logger.debug(
                            f"ID à supprimer (par document_title): {all_docs['ids'][i]} pour '{metadata['document_title']}'"
                        )
                        continue

                # Vérifier dans filename (si disponible)
                if "filename" in metadata and metadata["filename"]:
                    db_filename = metadata["filename"].lower()
                    if filename_lower == db_filename or filename_lower in db_filename:
                        chunk_ids_to_delete.append(all_docs["ids"][i])
                        logger.debug(
                            f"ID à supprimer (par filename): {all_docs['ids'][i]} pour '{metadata['filename']}'"
                        )
                        continue

                # Chercher dans le contenu du document
                if i < len(all_docs["documents"]) and all_docs["documents"][i]:
                    doc_content = all_docs["documents"][i].lower()
                    if filename_lower in doc_content:
                        # Vérifier que c'est bien une mention significative du fichier
                        if (
                            f"document: " in doc_content
                            and filename_lower
                            in doc_content.split("document: ")[1].split("\n")[0].lower()
                        ):
                            chunk_ids_to_delete.append(all_docs["ids"][i])
                            logger.debug(
                                f"ID à supprimer (par contenu): {all_docs['ids'][i]}"
                            )

            if not chunk_ids_to_delete:
                logger.warning(f"Aucun document trouvé avec le nom '{filename}'")
                return False

            # Supprimer tous les chunks trouvés
            logger.info(
                f"Suppression de {len(chunk_ids_to_delete)} chunks associés au document '{filename}'"
            )
            self.collection.delete(ids=chunk_ids_to_delete)

            logger.info(f"Document '{filename}' supprimé avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du document: {e}")
            return False

    def add_documents(
        self,
        chunks: List[Dict],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to ChromaDB with their embeddings

        Args:
            chunks: List of document chunks (text content)
            metadata: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not chunks:
            logger.warning(
                "Tentative d'ajout de documents avec une liste de chunks vide"
            )
            return

        logger.info(f"Ajout de {len(chunks)} documents à ChromaDB")

        # Generate embeddings
        logger.debug("Génération des embeddings pour les chunks")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embeddings_manager.get_embeddings(texts)
        logger.debug(f"Embeddings générés: {len(embeddings)}")

        # Use metadata from chunks if not provided externally
        if metadata is None:
            logger.debug("Utilisation des métadonnées contenues dans les chunks")
            metadata = [
                chunk.get("metadata", {"source": "document"}) for chunk in chunks
            ]

        # Ensure each metadata dict has at least one attribute
        logger.debug("Vérification des métadonnées")
        for i, meta in enumerate(metadata):
            if not meta:
                metadata[i] = {"source": "document"}

        # Generate IDs if not provided
        if ids is None:
            logger.debug("Génération d'IDs UUID pour les documents")
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Add to ChromaDB
        logger.debug("Ajout des documents à ChromaDB")
        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids
        )
        logger.info(
            f"Documents ajoutés avec succès (collection: {self.collection.name})"
        )

    def search(
        self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents. Handles both summarized and non-summarized chunks intelligently.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of results with documents and metadata
        """
        logger.info(
            f"Recherche dans ChromaDB: '{query}' (n_results={n_results}, filtres={filter_metadata})"
        )

        # Generate query embedding
        logger.debug("Génération de l'embedding pour la requête")
        query_embedding = self.embeddings_manager.get_embeddings([query])[0]

        # Search in ChromaDB
        logger.debug(f"Requête dans la collection '{self.collection.name}'")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # On récupère plus de résultats pour le post-traitement
            where=filter_metadata,
        )

        # Format and process results
        logger.debug(
            f"Formatage des résultats ({len(results['ids'][0])} documents trouvés)"
        )
        formatted_results = []
        seen_originals = set()  # Pour suivre les contenus originaux déjà vus

        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            is_summary = metadata.get("is_summary", "false").lower() == "true"
            original_content = metadata.get("original_content", "")

            # Si c'est un résumé, on ajoute le contenu original aux métadonnées pour référence
            if is_summary:
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i],
                    "is_summary": True,
                    "original_content": original_content
                }
                formatted_results.append(result)
                seen_originals.add(original_content)

            # Si c'est un contenu original et qu'on n'a pas déjà son résumé
            elif results["documents"][0][i] not in seen_originals:
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i],
                    "is_summary": False
                }
                formatted_results.append(result)

        # Trier par score de similarité et limiter au nombre demandé
        formatted_results.sort(key=lambda x: x["distance"])
        formatted_results = formatted_results[:n_results]

        logger.info(f"Recherche terminée, {len(formatted_results)} résultats")
        # Log des statistiques sur les résumés vs originaux
        summaries = sum(1 for r in formatted_results if r.get("is_summary", False))
        logger.info(f"Répartition : {summaries} résumés, {len(formatted_results) - summaries} originaux")
        
        return formatted_results

    def delete_collection(self) -> None:
        """Delete the current collection"""
        logger.warning(f"Suppression de la collection '{self.collection.name}'")
        self.client.delete_collection(self.collection.name)
        logger.info(f"Collection '{self.collection.name}' supprimée")

    def reset(self) -> None:
        """Reset the database"""
        logger.warning("Réinitialisation complète de la base de données ChromaDB")
        self.client.reset()
        logger.info("Base de données réinitialisée")

    def get_all_documents(self) -> Dict:
        """
        Récupère tous les documents de la base de données avec leurs métadonnées

        Returns:
            Dict: Dictionnaire avec les IDs des documents comme clés et leurs métadonnées comme valeurs
        """
        logger.debug("Récupération de tous les documents")

        try:
            # Récupérer tous les documents
            all_docs = self.collection.get()

            if not all_docs or "metadatas" not in all_docs or not all_docs["metadatas"]:
                logger.debug("Aucun document trouvé dans la base de données")
                return {}

            # Créer un dictionnaire avec les ID comme clés et les métadonnées comme valeurs
            result = {}
            for i, doc_id in enumerate(all_docs["ids"]):
                result[doc_id] = all_docs["metadatas"][i]

            logger.debug(f"Récupération terminée - {len(result)} documents trouvés")
            return result

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des documents: {e}")
            return {}
