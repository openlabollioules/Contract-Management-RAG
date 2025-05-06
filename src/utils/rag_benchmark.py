import os
import sys
from pathlib import Path
import json
import time
import shutil
from typing import List, Dict, Tuple
from datetime import datetime
from itertools import product
import pandas as pd
from tqdm import tqdm
import threading

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from core.document_manager import document_exists, cleanup_flag_documents
from core.chunk_summarizer import ChunkSummarizer
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from document_processing.llm_chat import LLMChat
from document_processing.text_chunker import TextChunker
from document_processing.reranker import Reranker
from core.graph_manager import GraphManager
from core.interaction import load_or_build_graph, get_graph_augmented_results, merge_results
from utils.logger import setup_logger
from document_processing.pdf_extractor import extract_pdf_text

# Configure logger
logger = setup_logger(__file__)

# Test questions and expected answers
TEST_CASES = [
    {
        "question": "Peux-tu m'indiquer les dates clés du Contrat A ?",
        "expected_answer": """
        Voici les dates clés du Contrat A présent dans le fichier :
        • Date de signature du contrat : 20 mars 2012.
        • Date de commencement des travaux : correspond à la date de signature, soit le 20 mars 2012.
        • Délai de 30 jours après le début du contrat : période prévue pour remplir certaines conditions supplémentaires.
        • Délais de livraison : précisés dans l'Annexe 2 ("Delivery Schedule") pour chaque composant et document critique.
        • Délai d'approbation des documents : 15 jours pour retour du client, 10 jours pour resoumission après commentaires.
        """
    },
    {
        "question": "Peux-tu me lister les éléments du contrat A qui impliquent le paiement potentiel d'indemnités ou de pénalités de la part du fournisseur ?",
        "expected_answer": """
        Voici les éléments du Contrat A qui impliquent un paiement potentiel d'indemnités ou de pénalités par le fournisseur :
        1. Retard de livraison d'équipement :
        • Des pénalités journalières ("Equipment Delay Liquidated Damages") sont appliquées si le fournisseur dépasse les délais de livraison contractuels :
        • 0,03 % du prix du composant/jour pendant le 1er mois,
        • 0,04 % pendant le 2e mois,
        • 0,05 % à partir du 3e mois.
        • Ces pénalités sont plafonnées à 10 % du prix du composant concerné.
        2. Retard dans la remise de documentation critique :
        • Pénalités de 500 € par document et par semaine de retard,
        • Plafond global fixé à 2 % du prix de l'unité concernée.
        3. Non-respect des garanties de performance :
        • Des pénalités peuvent s'appliquer si les performances garanties ne sont pas atteintes et que cela entraîne une demande du client final, sauf si la cause est externe au fournisseur.
        4. Indemnisation pour dommages :
        • Le fournisseur est responsable des dommages causés par des défauts de conception, fabrication ou négligence,
        • Il couvre également les réclamations liées à des violations de droits de propriété intellectuelle.
        5. Défauts non corrigés :
        • En cas de défaut non réparé dans un délai raisonnable après notification écrite, le client peut faire exécuter les réparations par un tiers aux frais du fournisseur.
        6. Résiliation pour retard :
        • Si le fournisseur atteint le plafond des pénalités pour retard et ne respecte pas un délai de livraison additionnel, le contrat peut être résilié unilatéralement par le client.
        """
    },
    {
        "question": "Dans le contrat A, quelle est la clause qui est la plus problématique du point de vue du fournisseur et pourquoi ?",
        "expected_answer": """
        La clause la plus problématique pour le fournisseur dans le Contrat A est probablement la suivante :

        ⚠️ Clause 11 : Pénalités de Retard sur Livraison d'Équipement
        📌 Pourquoi cette clause est problématique :
            • Cumul des pénalités : Jusqu'à 0,05 %/jour à partir du 3e mois de retard.
            • Plafond élevé : Plafonné à 10 % du prix de chaque composant.
            • Application automatique : Sans obligation de démonstration d'un préjudice réel de la part de l'acheteur.
            • Pénalités cumulables avec d'autres types de pénalités (ex. : documentation, performance) – ce qui peut générer une charge financière importante.

        🎯 Risques concrets pour le fournisseur :
            • Un simple retard logistique ou administratif peut entraîner des pénalités disproportionnées.
            • Aucune clause de grâce, de mise en demeure préalable ni de prise en compte de circonstances atténuantes (hors force majeure explicite).
        """
    },
    {
        "question": "Dans le contrat A, quel est le risque de change introduit par le fait qu'une partie des prix soient établis en roubles ?",
        "expected_answer": """
        Le Contrat A introduit un risque de change important du fait qu'une partie des prix soit libellée en roubles russes (RUB), comme cela est mentionné à l'article 7.1, qui distingue :
            • une partie du prix en euros (EUR), et
            • une partie du prix en roubles (RUB).

        ⚠️ Nature du risque de change pour le fournisseur :
            1. Volatilité du rouble : Le rouble est une monnaie soumise à une forte instabilité politique et économique. Sa valeur peut fluctuer brutalement, notamment en raison de :
                ◦ sanctions internationales,
                ◦ instabilités macroéconomiques,
                ◦ décisions monétaires unilatérales de la Russie.
            2. Risque de perte de marge : Si le fournisseur supporte des coûts en euros ou dollars mais facture en roubles, une dépréciation du rouble entre la signature et le paiement réduira significativement la valeur réelle perçue.
            3. Absence apparente de clause d'ajustement : Le contrat ne semble pas comporter de mécanisme de couverture ou d'indexation en cas de variation du taux de change, ce qui laisse le fournisseur totalement exposé.
        """
    },
    {
        "question": "Quelles sont les lois applicables mentionnées dans le contrat A ?",
        "expected_answer": """
        Le Contrat A précise la loi applicable dans l'article 24.1, comme suit :
        ⚖️ Loi applicable :
        Le contrat est régi par le droit matériel suisse, à l'exclusion de ses règles de conflit de lois.
        La Convention de Vienne de 1980 sur la vente internationale de marchandises ne s'applique pas.
        
        🧑‍⚖️ Règlement des litiges (Article 24.2) :
        • En cas de litige non résolu à l'amiable :
        • Il sera soumis à l'arbitrage selon les règles de la Chambre de commerce internationale (CCI).
        • L'arbitrage se tiendra à Genève, en langue anglaise.
        • La décision arbitrale est finale et contraignante pour les deux parties
        """
    },
    {
        "question": "A partir du contrat A, peux-tu dresser la liste des actions à mener par le fournisseur en termes de documents à fournir au client ?",
        "expected_answer": """
        Voici la liste des actions documentaires à mener par le fournisseur selon le Contrat A, ainsi que leurs modalités :
        📋 1. Livrables documentaires identifiés dans l'Annexe 2
        • Le fournisseur doit fournir tous les documents listés dans l'Exhibit 2 ("Document Delivery Schedule").
        • Cela inclut des documents techniques, qualité, essais, manuels de montage, mise en service, maintenance, etc.

        📝 2. Documentation nécessitant validation du client
        • Certains documents nécessitent revue, approbation ou acceptation du client. Cela est également spécifié dans l'Exhibit 2.
        • Le client doit retourner les documents sous 15 jours avec commentaires.
        • Le fournisseur doit soumettre une version corrigée sous 10 jours.

        🏷️ 3. Format, codification et transmission
        • Le format, codification, méthode d'envoi sont précisés dans un Supplementary Agreement à conclure dans les 90 jours suivant la signature.

        ⚠️ 4. Documentation critique
        • Toute "Critical Documentation" doit être livrée aux dates de l'Exhibit 2.
        • En cas de retard, des pénalités de 500 € par document/semaine peuvent être appliquées, plafonnées à 2 % du prix de l'unité.

        🌐 5. Langue
        • Toute la documentation doit être fournie en anglais.

        🚨 6. Correction des défauts
        • En cas de défaut ou d'omission dans un document, le client peut exiger une correction immédiate et diligente.
        """
    },
    {
        "question": "Quelles obligations du contrat A doivent être impérativement intégrées aux contrats qu'ALSTOM signera avec ses fournisseurs ou sous-traitants ?",
        "expected_answer": """
        Voici les obligations du Contrat A qui doivent impérativement être répercutées par ALSTOM dans ses contrats avec ses fournisseurs et sous-traitants (obligations dites "flow-down") :

        🔐 1. Engagements de confidentialité
        • Clause très stricte imposant la confidentialité pour une durée de 10 ans après divulgation.
        • Toute sous-traitance impliquant l'accès à des informations sensibles doit être encadrée par des engagements similaires.

        📄 2. Livraison de documentation critique
        • Le fournisseur principal est responsable de livrer les documents critiques à des dates fixes sous peine de pénalités de 500 €/document/semaine, plafonnées à 2 % du prix de l'unité.
        • Ces échéances doivent être transmises aux sous-traitants avec engagement contractuel ferme sur le respect des dates.

        ⏱️ 3. Délais de livraison et pénalités
        • Retards sur les composants entraînent des pénalités croissantes (jusqu'à 0,05 %/jour) avec un plafond de 10 % du prix du composant.
        • Les sous-traitants livrant des composants critiques doivent se voir imposer des pénalités similaires pour permettre au fournisseur principal de se retourner contre eux si besoin.

        ⚙️ 4. Garantie / responsabilité pour défauts
        • Obligation pour le fournisseur de réparer ou remplacer les composants défectueux pendant la période de garantie (jusqu'à 24 mois après acceptation).
        • ALSTOM doit s'assurer que ses fournisseurs offrent une garantie équivalente, avec droits de recours en cas de défaillance.

        🛑 5. Clause de non-responsabilité du client pour l'installation
        • Le Contrat A précise que le fournisseur ne sera pas tenu responsable des défauts liés à l'installation faite par des tiers. Si ALSTOM sous-traite l'installation, elle doit s'assurer que les responsabilités sont contractuellement bien réparties entre les acteurs concernés.

        🔄 6. Garanties bancaires
        • Obligation de fournir :
        • Garantie de remboursement d'acompte (100 % APBG),
        • Garantie de bonne exécution (5 % PBG),
        • Délai précis de remise (20 jours après démarrage).
        • Ces exigences doivent être imposées aux fournisseurs ou bancarisées à leur nom, si nécessaire.

        📦 7. Obligations en cas de résiliation
        • En cas de résiliation du contrat principal, ALSTOM peut devoir interrompre, transférer ou réclamer le matériel et les prestations.
        • Elle doit prévoir des clauses de transfert de propriété anticipée et de continuité avec ses fournisseurs pour couvrir ce risque.
        """
    }
]

class RAGBenchmark:
    def __init__(self):
        # Test configurations
        self.models = ["llama3.2:latest"]  # Your current Ollama models
        self.rerank_models = [
            None,  # No reranking
            "NV-RerankQA-Mistral-4B-v3",
            "bge-reranker-large",
            "Jina-ColBERT-v1"
        ]
        self.top_k_values = [3, 5, 7, 10]
        self.temperature_values = [0.1, 0.3, 0.5, 0.7]
        self.similarity_thresholds = [0.5, 0.6, 0.7, 0.8]
        self.summarize_options = [True, False]
        self.chat_modes = ["normal", "graph"]  # Test both normal chat and graph-enhanced chat
        
        # Paths
        self.workspace_dir = Path(__file__).resolve().parent.parent.parent
        self.contract_dir = self.workspace_dir / "data" / "Contract"
        self.db_dir = self.workspace_dir / "chroma_db"
        
        # Verify directories exist
        self._verify_paths()
        
        # Initialize results storage
        self.results = []
        self.results_lock = threading.Lock()
        
        # Create timestamp and results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.workspace_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize results files
        self.results_file = self.results_dir / f"rag_benchmark_{self.timestamp}.json"
        self.analysis_file = self.results_dir / f"rag_benchmark_analysis_{self.timestamp}.txt"
        self.excel_file = self.results_dir / f"rag_benchmark_detailed_{self.timestamp}.xlsx"
        
        # Initialize files with headers
        self._initialize_results_files()
        
        # Start analysis update thread
        self.stop_analysis_thread = False
        self.analysis_thread = threading.Thread(target=self._update_analysis_periodically)
        self.analysis_thread.start()
        
        # Initialize text chunker
        self.text_chunker = TextChunker()
        
        # Initialize reranker dict
        self.rerankers = {}

    def _verify_paths(self):
        """Verify all required paths exist"""
        if not self.contract_dir.exists():
            raise FileNotFoundError(f"Contract directory not found: {self.contract_dir}")
            
        # Get all supported files from contract directory
        self.contract_files = self._get_supported_files()
        if not self.contract_files:
            raise FileNotFoundError("No supported files found in contract directory")

    def _initialize_database(self, use_summarize: bool = False, rerank_model: str = None) -> None:
        """Initialize the vector database with contract files"""
        logger.info(f"Initializing database (summarize={use_summarize}, rerank={rerank_model})")
        
        try:
            # Create database directory if it doesn't exist
            self.db_dir.mkdir(mode=0o777, exist_ok=True)
            
            # Initialize components
            self.embeddings_manager = TextVectorizer()
            
            # Initialize vector database with a unique collection name for this configuration
            collection_name = f"contracts_{use_summarize}_{rerank_model if rerank_model else 'no_rerank'}"
            collection_name = collection_name.replace('-', '_').replace('/', '_')  # Sanitize name
            self.vector_db = VectorDBInterface(
                self.embeddings_manager,
                persist_directory=str(self.db_dir),
                collection_name=collection_name
            )
            
            # Reset the collection to start fresh
            self.vector_db.reset()
            
            # Initialize reranker if specified
            if rerank_model:
                if rerank_model not in self.rerankers:
                    self.rerankers[rerank_model] = Reranker(rerank_model)
            
            # Process each contract file
            for file in self.contract_files:
                file_path = self.contract_dir / file
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                logger.info(f"Processing {file}")
                
                # Get text content
                content = self._extract_text(file_path)
                if not content:  # Skip if no content was extracted
                    continue
                
                # Create chunks using TextChunker
                chunks = self.text_chunker.chunk_text(content, str(file_path))
                
                # Convert chunks to the expected format
                formatted_chunks = []
                for chunk in chunks:
                    # Create metadata dictionary
                    metadata = {
                        "section_number": str(getattr(chunk, "section_number", "unknown")),
                        "hierarchy": " -> ".join(chunk.hierarchy) if getattr(chunk, "hierarchy", None) else "unknown",
                        "document_title": str(getattr(chunk, "document_title", "unknown")),
                        "parent_section": str(getattr(chunk, "parent_section", "unknown")),
                        "chapter_title": str(getattr(chunk, "chapter_title", "unknown")),
                        "position": str(getattr(chunk, "position", "0")),
                        "total_chunks": str(getattr(chunk, "total_chunks", "0")),
                    }
                    
                    # Format the content with metadata
                    content = f"""
Section: {metadata['section_number']}
Hiérarchie complète: {metadata['hierarchy']}
Document: {metadata['document_title']}
Position: {metadata['position']}/{metadata['total_chunks']}

Contenu:
{chunk.content}
"""
                    formatted_chunks.append({"content": content, "metadata": metadata})
                
                # Apply summarization if enabled
                if use_summarize and formatted_chunks:
                    summarizer = ChunkSummarizer()
                    formatted_chunks = summarizer.summarize_chunks(formatted_chunks)
                
                # Add to database if we have chunks
                if formatted_chunks:
                    self.vector_db.add_documents(formatted_chunks)
                else:
                    logger.warning(f"No chunks were created for {file}")
            
            # Build graph
            self.graph = load_or_build_graph(self.vector_db, self.embeddings_manager)
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from a document file"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                text, _ = extract_pdf_text(file_path)
                return text
            elif suffix in ['.doc', '.docx']:
                from document_processing.doc_reader import extract_doc_text
                text, _ = extract_doc_text(file_path)
                return text
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def _initialize_results_files(self):
        """Initialize results files with headers"""
        # Initialize JSON results file
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        # Initialize analysis file with header
        with open(self.analysis_file, "w", encoding="utf-8") as f:
            f.write("# RAG Benchmark Results\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nAnalysis will be updated in real-time as tests progress...\n")

    def _update_analysis_periodically(self):
        """Update analysis file periodically with current results"""
        while not self.stop_analysis_thread:
            with self.results_lock:
                if self.results:  # Only update if we have results
                    self._generate_analysis()
            time.sleep(60)  # Update every minute

    def _generate_analysis(self):
        """Generate and save analysis based on current results"""
        try:
            df = pd.DataFrame(self.results)
            
            analysis = []
            analysis.append("# RAG Benchmark Results\n")
            analysis.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Overall statistics
            analysis.append("\n## Overall Statistics\n")
            analysis.append(f"Total configurations tested: {len(self.results)}")
            analysis.append(f"Total questions: {len(TEST_CASES)}")
            
            # Best configurations by metric
            metrics = ["accuracy", "relevance", "response_time", "total_time"]
            for metric in metrics:
                if metric in df.columns:
                    analysis.append(f"\n## Best Configuration for {metric}\n")
                    best_idx = df[metric].idxmax()
                    best_config = df.iloc[best_idx]
                    analysis.append(f"- Model: {best_config['model']}")
                    analysis.append(f"- Rerank Model: {best_config['rerank_model']}")
                    analysis.append(f"- Top K: {best_config['top_k']}")
                    analysis.append(f"- Temperature: {best_config['temperature']}")
                    analysis.append(f"- Similarity Threshold: {best_config['similarity_threshold']}")
                    analysis.append(f"- Summarize: {best_config['use_summarize']}")
                    analysis.append(f"- Chat Mode: {best_config['chat_mode']}")
                    analysis.append(f"- Score: {best_config[metric]:.3f}")
            
            # Model comparison
            if len(df) > 0:
                analysis.append("\n## Model Comparison\n")
                model_metrics = df.groupby(["model", "rerank_model", "chat_mode", "use_summarize"])[metrics].mean()
                analysis.append(model_metrics.to_string())
                
                # Performance analysis
                analysis.append("\n## Performance Analysis\n")
                perf_metrics = df.groupby(["model", "rerank_model"])[["response_time", "total_time"]].mean()
                analysis.append(perf_metrics.to_string())
            
            # Save analysis
            with open(self.analysis_file, "w", encoding="utf-8") as f:
                f.write("\n".join(analysis))
            
            # Update Excel file with detailed analysis
            with pd.ExcelWriter(self.excel_file) as writer:
                df.to_excel(writer, sheet_name="Raw Results", index=False)
                if len(df) > 0:
                    model_metrics.to_excel(writer, sheet_name="Model Comparison")
                    perf_metrics.to_excel(writer, sheet_name="Performance Analysis")
            
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")

    def _save_result(self, result: Dict):
        """Save a single result and update analysis"""
        with self.results_lock:
            self.results.append(result)
            
            # Update JSON file
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # Update analysis if we have enough results
            if len(self.results) % 10 == 0:  # Update every 10 results
                self._generate_analysis()

    def _calculate_metrics(self, response: Dict, question: str, expected_answer: str) -> Dict:
        """Calculate evaluation metrics for a response"""
        # TODO: Implement more sophisticated metrics
        # For now, we'll use simple presence/absence metrics
        answer = response.get("answer", "").lower()
        expected = expected_answer.lower()
        
        # Basic accuracy: check if key terms from expected answer are in the response
        key_terms = set(expected.split())
        found_terms = sum(1 for term in key_terms if term in answer)
        accuracy = found_terms / len(key_terms) if key_terms else 0.0
        
        # Basic relevance: check if the response uses the context
        relevance = 1.0 if response.get("context") else 0.0
        
        return {
            "accuracy": accuracy,
            "relevance": relevance
        }

    def run_single_test(
        self,
        model: str,
        rerank_model: str,
        top_k: int,
        temperature: float,
        similarity_threshold: float,
        use_summarize: bool,
        chat_mode: str,
        test_case: Dict
    ) -> Dict:
        """Run a single test with given parameters"""
        try:
            # Initialize chat interface
            llm_chat = LLMChat(model=model)
            
            # Record start time
            start_time = time.time()
            
            # Get response based on chat mode
            if chat_mode == "graph":
                response_from_llm_logic = get_graph_augmented_results(
                    self.vector_db,
                    self.graph,
                    test_case["question"],
                    top_k=top_k,
                    temperature=temperature,
                    similarity_threshold=similarity_threshold
                )
            else: # Normal mode
                # 1. Retrieve context from vector_db
                context_docs = self.vector_db.search(
                    query=test_case["question"],
                    n_results=top_k 
                )

                # 2. Filter by similarity_threshold 
                # (Chroma distance is 1-similarity for cosine, so distance <= 1-sim_thresh)
                final_context_docs = [
                    doc for doc in context_docs if doc['distance'] <= (1.0 - similarity_threshold)
                ]

                # 3. Prepare context string for the prompt
                if final_context_docs:
                    context_str = "\n\n---\n\n".join(
                        [f"Source {i+1} (distance: {doc['distance']:.4f}):\n{doc['document']}" for i, doc in enumerate(final_context_docs)]
                    )
                else:
                    context_str = "Aucun contexte pertinent trouvé après filtrage."

                # 4. Construct prompt for LLM
                prompt_template = """Tu es un assistant spécialisé dans l'analyse de contrats.
Voici le contexte pertinent extrait des documents. Utilise ce contexte pour répondre à la question.
Si l'information n'est pas dans le contexte, indique-le clairement.

Contexte:
{context}

Question: {question}

Réponse:"""
                prompt = prompt_template.format(context=context_str, question=test_case["question"])

                # 5. Call LLM and measure response time
                llm_call_start_time = time.time()
                llm_response_text = llm_chat.generate(
                    prompt,
                    options={"temperature": temperature} # Pass temperature via options
                )
                llm_call_end_time = time.time()
                llm_response_duration = llm_call_end_time - llm_call_start_time

                # 6. Reconstruct the response dictionary
                response_from_llm_logic = {
                    "answer": llm_response_text,
                    "context": [doc['document'] for doc in final_context_docs], # List of document strings
                    "response_time": llm_response_duration 
                }
            
            # Apply reranking if specified
            # Ensure context exists and is a list of strings before reranking
            if rerank_model and response_from_llm_logic and "context" in response_from_llm_logic and isinstance(response_from_llm_logic["context"], list) and response_from_llm_logic["context"]:
                reranker = self.rerankers[rerank_model]
                # Reranker expects a list of document strings
                reranked_context_strings = reranker.rerank(
                    test_case["question"],
                    response_from_llm_logic["context"], 
                    top_k=top_k 
                )
                response_from_llm_logic["context"] = reranked_context_strings # Update with reranked context
            
            # Calculate times
            end_time = time.time()
            total_time = end_time - start_time
            response_time = response_from_llm_logic.get("response_time", 0.0)
            
            # Calculate metrics
            metrics = self._calculate_metrics(response_from_llm_logic, test_case["question"], test_case["expected_answer"])
            
            result = {
                "model": model,
                "rerank_model": rerank_model,
                "top_k": top_k,
                "temperature": temperature,
                "similarity_threshold": similarity_threshold,
                "use_summarize": use_summarize,
                "chat_mode": chat_mode,
                "question": test_case["question"],
                "expected_answer": test_case["expected_answer"],
                "actual_answer": response_from_llm_logic.get("answer", ""),
                "context_used": response_from_llm_logic.get("context", ""),
                "response_time": response_time,
                "total_time": total_time,
                **metrics
            }
            
            # Save result immediately
            self._save_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in test: {str(e)}")
            return None

    def run_benchmark(self):
        """Run the complete benchmark with all configurations"""
        logger.info("Starting comprehensive RAG benchmark...")
        
        # Create all possible combinations of parameters
        configs = list(product(
            self.models,
            self.rerank_models,
            self.top_k_values,
            self.temperature_values,
            self.similarity_thresholds,
            self.summarize_options,
            self.chat_modes,
            TEST_CASES
        ))
        
        total_configs = len(configs)
        logger.info(f"Total configurations to test: {total_configs}")
        
        # Run tests
        with tqdm(total=total_configs, desc="Running tests") as pbar:
            for config in configs:
                (model, rerank, top_k, temp, sim_thresh, summarize, chat_mode, test_case) = config
                
                # Reinitialize database for each summarize/rerank combination
                self._initialize_database(use_summarize=summarize, rerank_model=rerank)
                
                result = self.run_single_test(
                    model=model,
                    rerank_model=rerank,
                    top_k=top_k,
                    temperature=temp,
                    similarity_threshold=sim_thresh,
                    use_summarize=summarize,
                    chat_mode=chat_mode,
                    test_case=test_case
                )
                
                if result:
                    pbar.update(1)
        
        # Stop analysis thread
        self.stop_analysis_thread = True
        self.analysis_thread.join()
        
        # Final analysis update
        self._generate_analysis()
        
        logger.info(f"Results saved to {self.results_file}")
        logger.info(f"Analysis saved to {self.analysis_file}")
        logger.info(f"Detailed Excel report saved to {self.excel_file}")

    def _get_supported_files(self) -> List[str]:
        """Get all supported files from contract directory"""
        supported_extensions = {'.pdf', '.doc', '.docx'}
        files = []
        
        for file in self.contract_dir.iterdir():
            if file.is_file() and file.suffix.lower() in supported_extensions:
                files.append(file.name)
                
        logger.info(f"Found {len(files)} supported files in contract directory")
        return files

if __name__ == "__main__":
    try:
        # Clean up any incorrectly stored flags
        cleanup_flag_documents()
        
        # Start the benchmark
        print("🚀 Starting RAG benchmark...")
        benchmark = RAGBenchmark()
        benchmark.run_benchmark()
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        sys.exit(1) 