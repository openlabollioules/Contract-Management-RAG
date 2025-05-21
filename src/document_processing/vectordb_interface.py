import uuid
from typing import Dict, List, Optional
from pathlib import Path
import os
import re

import chromadb
from chromadb.config import Settings

from utils.logger import setup_logger
from core.llm_service import LLMService

from .text_vectorizer import TextVectorizer

# Configurer le logger pour ce module
logger = setup_logger(__file__)


class VectorDBInterface:
    def __init__(
        self,
        embeddings_manager: TextVectorizer,
        persist_directory: str = "chroma_db",
        collection_name: str = "contracts",
        llm_service = None
    ):
        """
        Initialize ChromaDB manager with an embeddings manager

        Args:
            embeddings_manager: Instance of TextVectorizer for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            llm_service: LLM service for advanced text analysis
        """
        self.embeddings_manager = embeddings_manager
        self.persist_directory = persist_directory
        self.llm_service = llm_service
        logger.info(
            f"Initialisation de VectorDBInterface (persist_directory={persist_directory}, collection={collection_name})"
        )

        # Ensure the persist directory exists with proper permissions
        persist_path = Path(persist_directory)
        if not persist_path.exists():
            persist_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        
        # Set permissions for all existing files and directories
        for item in persist_path.rglob('*'):
            try:
                item.chmod(0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions for {item}: {e}")
        
        # Set umask to ensure new files are created with proper permissions
        old_umask = os.umask(0)
        try:
            # Initialize ChromaDB client with settings
            logger.debug(f"Création du client ChromaDB persistant")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                ),
            )

            # Create or get collection
            logger.debug(f"Récupération ou création de la collection '{collection_name}'")
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{collection_name}' initialisée")
        finally:
            # Restore original umask
            os.umask(old_umask)

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

    def _extract_keywords(self, text: str) -> list:
        """
        Extract keywords from text using LLM analysis.
        Falls back to basic keyword detection if LLM is not available.

        Args:
            text: Text to analyze

        Returns:
            List of detected keywords
        """
        # If LLM service is not available, use fallback immediately
        if not self.llm_service or not self.llm_service.client:
            logger.debug("LLM service not available, using fallback keyword detection")
            return self._extract_keywords_fallback(text)

        try:
            # Prepare the prompt for the LLM
            prompt = f"""Analyze the following text and identify important keywords related to contracts, business terms, and legal concepts.
            Focus on terms that would be relevant for contract management and legal analysis.
            Return only the keywords, separated by commas.

            Text to analyze:
            {text}

            Keywords:"""

            # Get response from LLM
            response = self.llm_service.generate(prompt)
            
            # Process the response to get keywords
            if response:
                # Split by comma and clean up each keyword
                keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
                logger.debug(f"Keywords detected by LLM: {keywords}")
                return keywords
            else:
                logger.warning("No keywords detected by LLM, using fallback")
                return self._extract_keywords_fallback(text)

        except Exception as e:
            logger.error(f"Error in LLM keyword extraction: {str(e)}")
            # Fallback to basic keyword detection if LLM fails
            return self._extract_keywords_fallback(text)

    def _extract_keywords_fallback(self, text: str) -> list:
        """
        Fallback method for keyword extraction using predefined terms.
        Used when LLM analysis fails.

        Args:
            text: Text to analyze

        Returns:
            List of detected keywords without duplicates
        """
        # Dictionary to store terms and their translations
        terms_dict = {
            # Termes contractuels de base
            "montant": "amount",
            "échéance": "due date",
            "livraison": "delivery",
            "pénalité": "penalty",
            "date": "date",
            "contrat": "contract",
            "signature": "signature",
            "avenant": "amendment",
            "résiliation": "termination",
            "paiement": "payment",
            "garantie": "warranty",
            "obligation": "obligation",
            "clause": "clause",
            
            # Nouveaux termes contractuels
            "parties": "parties",
            "objet du contrat": "scope of the contract",
            "livraisons": "deliverables",
            "frais": "fees",
            "confidentialité": "confidentiality",
            "propriété intellectuelle": "intellectual property",
            "clause de résiliation": "termination clause",
            "conformité légale": "legal compliance",
            "support technique": "technical support",
            "responsabilités": "responsibilities",
            "délais": "milestones",
            "résolution des litiges": "dispute resolution",
            "force majeure": "force majeure",
            "accès aux données": "data access",
            "sécurité des données": "data security",
            "licence": "license",
            "exclusivité": "exclusivity",
            "durée du contrat": "contract term",
            "résiliation pour cause": "termination for cause"
        }

        # Set to store unique found terms
        found_terms = set()

        # Search for terms in text
        for term_fr, term_en in terms_dict.items():
            if re.search(rf"\b{term_fr}\b", text, re.IGNORECASE):
                found_terms.add(term_fr)
            if re.search(rf"\b{term_en}\b", text, re.IGNORECASE):
                found_terms.add(term_en)

        return list(found_terms)

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

        # Détecter les dates dans chaque chunk avant l'ajout à la base de données
        for chunk in chunks:
            # Détecter les dates dans le contenu du chunk
            dates = self._detect_dates(chunk.get('content', ''))
            if dates:
                # Ajouter les dates aux métadonnées du chunk
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                # Convertir la liste de dates en string pour ChromaDB
                chunk['metadata']['dates'] = '; '.join(dates)
            keywords = self._extract_keywords(chunk.get('content', ''))
            if keywords:
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                chunk['metadata']['keywords'] = ', '.join(keywords)

        # Afficher les chunks avec les dates détectées
        print_chunks_with_dates(chunks)

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

    def _detect_dates(self, text: str) -> List[str]:
        """
        Detect dates in text using regex patterns.
        Supports various date formats commonly found in contracts.

        Args:
            text: Text to analyze

        Returns:
            List of detected dates
        """
        # Common date patterns in contracts
        date_patterns = [
            # DD/MM/YYYY or DD-MM-YYYY
            r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b',
            # YYYY/MM/DD or YYYY-MM-DD
            r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',
            # Month DD, YYYY (e.g., "January 1, 2024")
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b',
            # DD Month YYYY (e.g., "1 January 2024")
            r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',
            # French date format (e.g., "le 1er janvier 2024")
            r'\ble\s+(?:0?[1-9]|[12][0-9]|3[01])(?:er|ème)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(?:19|20)\d{2}\b',
            # DD.MM.YYYY (e.g., "06.12.2007")
            r'\b(0?[1-9]|[12][0-9]|3[01])\.(0?[1-9]|1[0-2])\.(19|20)\d{2}\b',
            # Month DDth YYYY (e.g., "September 30th 2012")
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\s+(?:19|20)\d{2}\b',
            # DDth Month YYYY (e.g., "22nd August 2017")
            r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',
            # Month DDth, YYYY (e.g., "February 2nd, 2012")
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th),\s+(?:19|20)\d{2}\b'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                logger.debug(f"Date trouvée: {match.group(0)}")
                dates.append(match.group(0))

        return dates

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
        """Reset the database and recreate the collection"""
        logger.warning("Réinitialisation complète de la base de données ChromaDB")
        try:
            # Delete the collection if it exists
            try:
                self.client.delete_collection(self.collection.name)
            except Exception as e:
                logger.debug(f"Collection deletion failed (might not exist): {e}")
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Base de données réinitialisée et collection recréée")
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            # If something went wrong, try to ensure we have a valid collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )

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

    def select_context(self, results: List[Dict], query: str) -> List[Dict]:
        # Prioritize chunks with matching metadata
        relevant_chunks = []
        for result in results:
            if (result['metadata'].get('keywords') and 
                any(kw in query.lower() for kw in result['metadata']['keywords'].split(','))):
                relevant_chunks.append(result)
        
        return relevant_chunks or results  # Fallback to all results if no metadata matches

def print_chunks_with_dates(chunks):
    print("\n📝 Affichage des chunks avec métadonnées:")
    print("=" * 80)
    
    # Compteur total de mots-clés
    total_keywords = set()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}/{len(chunks)}")
        print("-" * 40)
        
        # Récupérer toutes les métadonnées
        metadata = chunk.get('metadata', {})
        
        # Afficher les métadonnées de base
        print(f"📄 Document: {metadata.get('document_title', 'Sans titre')}")
        print(f"📑 Section: {metadata.get('section_number', 'Non spécifiée')}")
        
        # Afficher les dates si présentes
        if metadata.get('dates'):
            print(f"\n📅 Dates détectées: {metadata['dates']}")
        
        # Afficher les mots-clés si présents
        if metadata.get('keywords'):
            keywords = metadata['keywords'].split(', ')
            total_keywords.update(keywords)
            print(f"\n🔑 Mots-clés trouvés dans ce chunk:")
            for kw in keywords:
                print(f"  • {kw}")
        
        # Afficher toutes les autres métadonnées
        other_metadata = {k: v for k, v in metadata.items() 
                         if k not in ['document_title', 'section_number', 'dates', 'keywords']}
        if other_metadata:
            print("\n📋 Autres métadonnées:")
            for key, value in other_metadata.items():
                print(f"  • {key}: {value}")
        
        print("\n📝 Contenu:")
        print(chunk.get('content', ''))
        print("-" * 40)
    
    # Afficher le résumé des mots-clés à la fin
    if total_keywords:
        print("\n📊 Résumé des mots-clés trouvés:")
        print("=" * 40)
        for kw in sorted(total_keywords):
            print(f"  • {kw}")
        print(f"\nTotal des mots-clés uniques: {len(total_keywords)}")
