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
        Detect dates in text using comprehensive regex patterns.
        Supports a wide variety of date formats commonly found in contracts and documents.

        Args:
            text: Text to analyze

        Returns:
            List of detected dates without duplicates
        """
        # Process the text to avoid duplicate matches from overlapping patterns
        # Apply patterns in order of specificity (most specific first) 
        # and avoid adding overlapping matches
        
        # Track matched spans to avoid overlapping matches
        matched_spans = []
        unique_dates = set()
        
        # Group patterns by priority (specific formats first, general formats last)
        pattern_groups = [
            # Group 1: Most specific patterns (with context words)
            [
                # Dates with specific words (e.g., "Dated: January 1, 2024")
                r'(?:dated|dated\s+as\s+of|as\s+of|effective\s+date|with\s+effect\s+from|en\s+date\s+du)[\s:]+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(?:19|20)\d{2}',
                r'(?:dated|dated\s+as\s+of|as\s+of|effective\s+date|with\s+effect\s+from|en\s+date\s+du)[\s:]+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}',
                
                # Date spans (commonly found in contracts)
                r'(?:from|du)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(?:19|20)\d{2}\s+(?:to|au|jusqu\'au)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(?:19|20)\d{2}',
                
                # Specific date references in contracts
                r'\bsigned\s+on\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(?:19|20)\d{2}\b',
                r'\bsigné\s+le\s+(?:0?[1-9]|[12][0-9]|3[01])(?:er|ère|ème|e|è)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|jan|fév|mar|avr|mai|juin|juil|août|sept|oct|nov|déc)\.?\s+(?:19|20)\d{2}\b',
                
                # French date format (e.g., "le 1er janvier 2024")
                r'\ble\s+(?:0?[1-9]|[12][0-9]|3[01])(?:er|ère|ème|e|è)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(?:19|20)\d{2}\b',
            ],
            
            # Group 2: Complete dates (day, month, year)
            [
                # Month DD, YYYY (e.g., "January 1, 2024")
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b',
                # DD Month YYYY (e.g., "1 January 2024")
                r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',
                # Month DDth YYYY (e.g., "September 30th 2012")
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\s+(?:19|20)\d{2}\b',
                # DDth Month YYYY (e.g., "22nd August 2017")
                r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',
                # Month DDth, YYYY (e.g., "February 2nd, 2012")
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th),\s+(?:19|20)\d{2}\b',
                # Abbreviated month formats (e.g., "Jan 1, 2024", "1 Jan 2024")
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b',
                r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:19|20)\d{2}\b',
                
                # French without "le" (e.g., "1er janvier 2024")
                r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:er|ère|ème|e|è)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(?:19|20)\d{2}\b',
                # French abbreviated months (e.g., "1er jan. 2024")
                r'\b(?:0?[1-9]|[12][0-9]|3[01])(?:er|ère|ème|e|è)?\s+(?:jan|fév|mar|avr|mai|juin|juil|août|sept|oct|nov|déc)\.?\s+(?:19|20)\d{2}\b',
            ],
            
            # Group 3: Numeric date formats
            [
                # DD/MM/YYYY or DD-MM-YYYY
                r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b',
                # YYYY/MM/DD or YYYY-MM-DD (ISO format)
                r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',
                # DD.MM.YYYY (European format with dots)
                r'\b(0?[1-9]|[12][0-9]|3[01])\.(0?[1-9]|1[0-2])\.(19|20)\d{2}\b',
                # MM.DD.YYYY (US format with dots)
                r'\b(0?[1-9]|1[0-2])\.(0?[1-9]|[12][0-9]|3[01])\.(19|20)\d{2}\b',
            ],
            
            # Group 4: Partial dates (month/year or quarter/year)
            [
                # Quarter and year references (common in contracts)
                r'\bQ[1-4]\s+(?:19|20)\d{2}\b',
                r'\b(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:19|20)\d{2}\b',
                r'\b(?:1er|2e|3e|4e|premier|deuxième|troisième|quatrième)\s+trimestre\s+(?:de\s+)?(?:19|20)\d{2}\b',
                
                # Month and year only
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:19|20)\d{2}\b',
                r'\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|jan|fév|mar|avr|mai|juin|juil|août|sept|oct|nov|déc)\.?\s+(?:19|20)\d{2}\b',
            ]
        ]
        
        # Process each pattern group in order
        for group_idx, pattern_group in enumerate(pattern_groups):
            for pattern in pattern_group:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get match span to check for overlaps
                    start, end = match.span()
                    date_str = match.group(0).strip()
                    
                    # Check if this match overlaps with any previously matched span
                    overlaps = False
                    for prev_start, prev_end in matched_spans:
                        # Check for overlap
                        if (start <= prev_end and end >= prev_start):
                            overlaps = True
                            # If this is a more specific pattern (earlier group), replace the previous match
                            if group_idx == 0:  # Most specific patterns group
                                # Remove any dates that might be contained in this span
                                dates_to_remove = []
                                for prev_date in unique_dates:
                                    if prev_date in date_str:
                                        dates_to_remove.append(prev_date)
                                        
                                for remove_date in dates_to_remove:
                                    unique_dates.remove(remove_date)
                                
                                # Add this more specific match
                                logger.debug(f"Date trouvée (spécifique): {date_str}")
                                unique_dates.add(date_str)
                                # Update the span
                                matched_spans.remove((prev_start, prev_end))
                                matched_spans.append((start, end))
                            break
                    
                    # If no overlap, add this match
                    if not overlaps:
                        logger.debug(f"Date trouvée: {date_str}")
                        unique_dates.add(date_str)
                        matched_spans.append((start, end))

        # Convert to list and sort for consistent output
        dates_list = sorted(list(unique_dates))
        
        # If dates were found, log them at a higher level
        if dates_list:
            logger.info(f"Dates détectées ({len(dates_list)}): {', '.join(dates_list)}")
            
        return dates_list

    def search(
        self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents. Handles both summarized and non-summarized chunks intelligently.
        Automatically detects date-related queries and enhances results with date-specific filtering.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of results with documents and metadata, prioritizing date-relevant content when appropriate
        """
        logger.info(
            f"Recherche dans ChromaDB: '{query}' (n_results={n_results}, filtres={filter_metadata})"
        )

        # Enhanced date-related terms detection - expanded with more terms
        date_related_terms = [
            # French terms
            'date', 'échéance', 'délai', 'terme', 'expire', 'expiration', 'calendrier', 
            'planning', 'horaire', 'jour', 'mois', 'année', 'trimestre', 'semestre',
            'période', 'durée', 'temps', 'chronologie', 'deadline', 'livraison', 
            'anniversaire', 'signature', 'préavis',
            # English terms
            'deadline', 'calendar', 'schedule', 'day', 'month', 'year', 'quarter', 
            'semester', 'period', 'duration', 'time', 'timeline', 'delivery', 
            'anniversary', 'signature', 'notice'
        ]
        
        is_date_query = any(term in query.lower() for term in date_related_terms)

        print(f" is date query : {is_date_query}")
        
        # Detect dates in the query - more aggressive detection for any query with numeric content
        dates_in_query = []
        date_filter_metadata = None
        
        # Check for digits that might be part of dates, or explicit date terms
        if is_date_query or any(char.isdigit() for char in query):
            logger.debug("Détection de dates dans la requête")
            dates_in_query = self._detect_dates(query)

            if dates_in_query:
                logger.info(f"Dates détectées dans la requête: {dates_in_query}")
                
                # Create a copy of filter_metadata to avoid modifying the original
                if filter_metadata is None:
                    date_filter_metadata = {}
                else:
                    date_filter_metadata = filter_metadata.copy()
                
                # Build date filter - improved to handle multiple dates more effectively
                if len(dates_in_query) == 1:
                    # For a single date, use a simple contains filter
                    date_filter_metadata["dates"] = {"$contains": dates_in_query[0]}
                else:
                    # For multiple dates, create a more optimized OR filter
                    date_conditions = []
                    for date in dates_in_query:
                        date_conditions.append({"dates": {"$contains": date}})
                    
                    # If there are existing $or conditions, we need to handle them properly
                    if "$or" in date_filter_metadata:
                        existing_or = date_filter_metadata["$or"]
                        date_filter_metadata["$or"] = existing_or + date_conditions
                    else:
                        date_filter_metadata["$or"] = date_conditions
        
        # Generate query embedding
        logger.debug("Génération de l'embedding pour la requête")
        query_embedding = self.embeddings_manager.get_embeddings([query])[0]
        
        # Perform two searches if date detection is active: one with date filter and one standard
        if dates_in_query:
            # First search with date filters
            logger.debug(f"Requête avec filtres de date dans la collection '{self.collection.name}'")
            date_filtered_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                where=date_filter_metadata,
            )
            
            # Second search without date filters (standard)
            logger.debug(f"Requête standard dans la collection '{self.collection.name}'")
            standard_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
            )
            
            # Combine results, prioritizing date-filtered ones
            combined_ids = []
            combined_docs = []
            combined_metadatas = []
            combined_distances = []
            
            # First add all date-filtered results
            if len(date_filtered_results["ids"]) > 0 and len(date_filtered_results["ids"][0]) > 0:
                combined_ids.extend(date_filtered_results["ids"][0])
                combined_docs.extend(date_filtered_results["documents"][0])
                combined_metadatas.extend(date_filtered_results["metadatas"][0])
                combined_distances.extend(date_filtered_results["distances"][0])
            
            # Then add standard results that aren't already included
            if len(standard_results["ids"]) > 0 and len(standard_results["ids"][0]) > 0:
                for i, doc_id in enumerate(standard_results["ids"][0]):
                    if doc_id not in combined_ids:
                        combined_ids.append(doc_id)
                        combined_docs.append(standard_results["documents"][0][i])
                        combined_metadatas.append(standard_results["metadatas"][0][i])
                        combined_distances.append(standard_results["distances"][0][i])
            
            # Create a results structure similar to what collection.query returns
            results = {
                "ids": [combined_ids],
                "documents": [combined_docs],
                "metadatas": [combined_metadatas],
                "distances": [combined_distances]
            }
            
            logger.info(f"Recherche combinée - {len(combined_ids)} résultats au total")
        else:
            # Standard search without date detection
            logger.debug(f"Requête standard dans la collection '{self.collection.name}'")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Récupère plus de résultats pour le post-traitement
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

            # Check if this result contains any of the dates from the query
            contains_query_date = False
            if dates_in_query and 'dates' in metadata:
                chunk_dates = metadata['dates'].split('; ')
                contains_query_date = any(date in chunk_dates for date in dates_in_query)

            # Si c'est un résumé, on ajoute le contenu original aux métadonnées pour référence
            if is_summary:
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i],
                    "is_summary": True,
                    "original_content": original_content,
                    "contains_query_date": contains_query_date
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
                    "is_summary": False,
                    "contains_query_date": contains_query_date
                }
                formatted_results.append(result)

        # Sort results, prioritizing documents with matching dates
        if dates_in_query:
            # First by date match, then by distance score
            formatted_results.sort(key=lambda x: (not x.get("contains_query_date", False), x["distance"]))
        else:
            # Sort only by distance score
            formatted_results.sort(key=lambda x: x["distance"])
        
        # Limit to requested number of results
        formatted_results = formatted_results[:n_results]

        logger.info(f"Recherche terminée, {len(formatted_results)} résultats")
        
        # Log des statistiques sur les résumés vs originaux
        summaries = sum(1 for r in formatted_results if r.get("is_summary", False))
        date_matches = sum(1 for r in formatted_results if r.get("contains_query_date", False))
        
        logger.info(f"Répartition : {summaries} résumés, {len(formatted_results) - summaries} originaux")
        if dates_in_query:
            logger.info(f"Résultats contenant des dates de la requête : {date_matches}/{len(formatted_results)}")
        
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
        """
        Select and prioritize the most relevant chunks based on query content.
        Especially handles date-related queries with advanced prioritization.
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            List of selected context chunks, prioritized based on relevance
        """
        # Expanded list of date-related terms for context selection
        date_related_terms = [
            # French terms
            'date', 'échéance', 'délai', 'terme', 'expire', 'expiration', 'calendrier', 
            'planning', 'horaire', 'jour', 'mois', 'année', 'trimestre', 'semestre',
            'période', 'durée', 'temps', 'chronologie', 'deadline', 'livraison', 
            'anniversaire', 'signature', 'préavis',
            # English terms
            'deadline', 'calendar', 'schedule', 'day', 'month', 'year', 'quarter', 
            'semester', 'period', 'duration', 'time', 'timeline', 'delivery', 
            'anniversary', 'signature', 'notice'
        ]
        
        is_date_query = any(term in query.lower() for term in date_related_terms)
        dates_in_query = self._detect_dates(query) if is_date_query or any(char.isdigit() for char in query) else []
        
        # If there are matching dates in the results, prioritize them but don't exclude other results
        date_relevant_chunks = []
        non_date_chunks = []
        
        # Separate date-relevant chunks from others
        for result in results:
            # Check if this result has dates that match the query
            date_relevant = False
            
            if dates_in_query:
                # First check "contains_query_date" flag set by the search method
                if result.get('contains_query_date', False):
                    date_relevant = True
                # Then check dates in metadata if the flag isn't set
                elif result['metadata'].get('dates'):
                    chunk_dates = result['metadata']['dates'].split('; ')
                    # Check for exact matches with any date in the query
                    if any(query_date in chunk_dates for query_date in dates_in_query):
                        date_relevant = True
                    # Also check for partial date matches (e.g., just month+year)
                    else:
                        for query_date in dates_in_query:
                            for chunk_date in chunk_dates:
                                # Extract components from dates for partial matching
                                if self._dates_partially_match(query_date, chunk_date):
                                    date_relevant = True
                                    break
                            if date_relevant:
                                break
            # For date-related queries without specific dates, check if the chunk has any dates
            elif is_date_query and result['metadata'].get('dates'):
                date_relevant = True
                
            # Add to appropriate list
            if date_relevant:
                date_relevant_chunks.append(result)
            else:
                non_date_chunks.append(result)
        
        # Check for keyword relevance in non-date chunks
        keyword_relevant_chunks = []
        other_chunks = []
        
        for result in non_date_chunks:
            # Check for keyword relevance
            keyword_relevant = False
            if result['metadata'].get('keywords') and any(
                kw.lower() in query.lower() 
                for kw in result['metadata']['keywords'].split(', ')):
                keyword_relevant = True
                keyword_relevant_chunks.append(result)
            else:
                other_chunks.append(result)
        
        # Sort each category by similarity
        date_relevant_chunks.sort(key=lambda x: x.get('distance', float('inf')))
        keyword_relevant_chunks.sort(key=lambda x: x.get('distance', float('inf')))
        other_chunks.sort(key=lambda x: x.get('distance', float('inf')))
        
        # Log statistics
        logger.info(f"Répartition des résultats:")
        logger.info(f"- Chunks avec dates pertinentes: {len(date_relevant_chunks)}")
        logger.info(f"- Chunks avec mots-clés pertinents: {len(keyword_relevant_chunks)}")
        logger.info(f"- Autres chunks (similarité sémantique): {len(other_chunks)}")
        
        # Combine results - first date chunks, then keyword chunks, then remaining chunks by similarity
        combined_results = []
        
        # Add date chunks first
        combined_results.extend(date_relevant_chunks)
        
        # Add keyword chunks that aren't already included
        combined_ids = {r['id'] for r in combined_results}
        for chunk in keyword_relevant_chunks:
            if chunk['id'] not in combined_ids:
                combined_results.append(chunk)
                combined_ids.add(chunk['id'])
        
        # Add remaining chunks by similarity until we reach the limit
        for chunk in other_chunks:
            if chunk['id'] not in combined_ids:
                combined_results.append(chunk)
                combined_ids.add(chunk['id'])
        
        # Limit to a reasonable number of results to avoid overwhelming the context
        max_results = min(20, len(combined_results))
        logger.info(f"Retournant {max_results} résultats combinés")
        
        # Return the combined and prioritized results
        return combined_results[:max_results]

    def _dates_partially_match(self, date1: str, date2: str) -> bool:
        """
        Check if two dates partially match (e.g., same month and year).
        
        Args:
            date1: First date string
            date2: Second date string
            
        Returns:
            True if dates partially match, False otherwise
        """
        # Extract common components that might be in dates
        months_fr = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                     'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
        months_en = ['january', 'february', 'march', 'april', 'may', 'june', 
                     'july', 'august', 'september', 'october', 'november', 'december']
        
        # Convert to lowercase for case-insensitive matching
        date1_lower = date1.lower()
        date2_lower = date2.lower()
        
        # Extract years - usually 4 digits
        year_pattern = r'(19|20)\d{2}'
        years1 = re.findall(year_pattern, date1_lower)
        years2 = re.findall(year_pattern, date2_lower)
        
        # Extract months - check for textual and numeric months
        # Check for French and English month names
        months1_fr = [m for m in months_fr if m in date1_lower]
        months2_fr = [m for m in months_fr if m in date2_lower]
        months1_en = [m for m in months_en if m in date1_lower]
        months2_en = [m for m in months_en if m in date2_lower]
        
        # Check for numeric months (1-12)
        numeric_month_pattern = r'\b(0?[1-9]|1[0-2])\b'
        numeric_months1 = re.findall(numeric_month_pattern, date1_lower)
        numeric_months2 = re.findall(numeric_month_pattern, date2_lower)
        
        # Match based on components:
        
        # 1. If both dates have years, check if any year matches
        if years1 and years2:
            year_match = any(y1 == y2 for y1 in years1 for y2 in years2)
            if not year_match:
                return False  # Years present but don't match
        else:
            # If only one has years, this is not a blocking factor
            year_match = True
        
        # 2. If both dates have textual months (in either language), check if any month matches
        month_match = False
        if (months1_fr and months2_fr):
            month_match = any(m1 == m2 for m1 in months1_fr for m2 in months2_fr)
        elif (months1_en and months2_en):
            month_match = any(m1 == m2 for m1 in months1_en for m2 in months2_en)
        # Cross-language matching (French to English)
        elif months1_fr and months2_en:
            for i, m_fr in enumerate(months_fr):
                if m_fr in months1_fr and months_en[i] in months2_en:
                    month_match = True
                    break
        # Cross-language matching (English to French)
        elif months1_en and months2_fr:
            for i, m_en in enumerate(months_en):
                if m_en in months1_en and months_fr[i] in months2_fr:
                    month_match = True
                    break
        # Numeric month matching
        elif numeric_months1 and numeric_months2:
            month_match = any(m1 == m2 for m1 in numeric_months1 for m2 in numeric_months2)
        
        # If months are present in both dates, they must match
        if ((months1_fr or months1_en or numeric_months1) and 
            (months2_fr or months2_en or numeric_months2)):
            if not month_match:
                return False
        
        # Check for explicit numeric day matching
        day_pattern = r'\b(0?[1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b'
        days1 = re.findall(day_pattern, date1_lower)
        days2 = re.findall(day_pattern, date2_lower)
        
        # If both have days, they should match for a partial match
        if days1 and days2:
            day_match = any(d1[0] == d2[0] for d1 in days1 for d2 in days2)
            if not day_match:
                return False
        
        # If we've made it here, the dates are considered to partially match
        # if at least one component (year or month) matched
        return year_match or month_match

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
