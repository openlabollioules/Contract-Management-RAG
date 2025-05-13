import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from typing import List, Dict, Optional, Any, Tuple
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from core.document_manager import cleanup_flag_documents
from core.chunk_summarizer import ChunkSummarizer
from document_processing.text_vectorizer import TextVectorizer
from document_processing.vectordb_interface import VectorDBInterface
from document_processing.llm_chat import LLMChat
from document_processing.text_chunker import TextChunker
from document_processing.reranker import Reranker
from core.graph_manager import GraphManager
from core.interaction import load_or_build_graph, get_graph_augmented_results, merge_results
from document_processing.pdf_extractor import extract_pdf_text
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__file__)

# Test questions and expected answers
TEST_CASES = [
    {
        "question": "Peux-tu m'indiquer les dates clés du Contrat A ?",
        "expected_answer": """• 2 février 2012 : signature du « Main Contract » n° 2012-KIT-002/2/07/2012 entre AAEM et CEPD, auquel le Contrat A est adossé.  
            • 20 mars 2012 : signature du présent « Equipment Supply Contract » (Contrat A) entre ALSTOM Power Systems SA et AAEM.  
            • 30 septembre 2012 (au plus tard) : signature du « Supplementary Agreement » devant formaliser tout ré-ajustement de prix - délai de 60 jours prévu à la clause 5.1.2 après constat d'une variation économique majeure.  
            • 22 août 2017 : date estimée de remise à l'essai et mise en service de l'Unité 1 (Exhibit 2 - Delivery Schedule).  
            • 22 août 2019 : date estimée de remise à l'essai et mise en service de l'Unité 2 (Exhibit 2 - Delivery Schedule)."""
    },
    {
        "question": "Peux-tu me lister les éléments du contrat A qui impliquent le paiement potentiel d'indemnités ou de pénalités de la part du fournisseur ?",
        "expected_answer": """Les pénalités contractuelles sont concentrées dans les articles 10, 12, 13 et 16 :
            • Art 13.1 - « Equipment Delay LD » : 0,20 % du prix du composant et par semaine de retard, plafonné à 10 % du prix de ce composant citeturn17file3.  
            • Art 13.2.1 - « Documentation Delay LD » : 500 €/document/semaine pour tout document critique livré hors délai citeturn17file13.  
            • Art 13.3.4 - « Key-Milestone LD » en cas de non-atteinte d'une étape clef citeturn17file11.  
            • Art 13.4 - « Performance Guarantee LD » : 0,13 % du prix contrat/MWe de déficit, avec un plafond global de 10 % du prix contrat citeturn19file2.  
            • Art 10.5 à 10.9 - re-tests, inspections et frais associés répétés à la charge du fournisseur.  
            • Art 10.14 / 10.17 / 10.18 - pénalités liées au respect des critères de puissance (1 197 MWe) et de rendement.  
            • Art 4.32 - 4.33 - pénalités documentaires supplémentaires (supplémentaires à l'art 13.2).  
            • Art 16.1 - plafond global de responsabilité fixé à 15 % du prix contrat citeturn19file0."""
    },
    {
        "question": "Dans le contrat A, quelle est la clause qui est la plus problématique du point de vue du fournisseur et pourquoi ?",
        "expected_answer": """La clause la plus lourde pour ALSTOM est l'Article 16 « Limits of Liability ».  
            - Elle plafonne la responsabilité cumulée du vendeur à 15 % du prix contrat (16.1 a & b), mais **exclut** de ce plafond les indemnités contractuelles (art 13) et les dommages nucléaires régis par la Convention de Vienne 1963, ce qui ouvre un risque financier bien supérieur, potentiellement illimité pour certains chefs de pertes citeturn19file0 citeturn19file6.  
            - Elle vient s'ajouter aux pénalités spécifiques de l'Article 13 ; cumulées, ces sommes peuvent dépasser la marge du fournisseur.  
            - Aucune contre-partie équivalente n'est prévue pour limiter la responsabilité du client (art 14 Compensation).  
            En pratique, cet article oblige le fournisseur à s'assurer ou à provisionner largement, ce qui dégrade fortement son bilan et son prix."""
    },
    {
        "question": "Dans le contrat A, quel est le risque de change introduit par le fait qu'une partie des prix soient établis en roubles ?",
        "expected_answer": """Le risque de change est contenu :  
            • Le prix contrat est établi et payé en euros (clau. 5.3.2), et seuls certains coûts locaux pourraient être supportés en roubles - ils sont répercutés via la formule d'indexation économique jointe à l'Exhibit 4.  
            • La clause 5.1.2 prévoit que toute variation économique ou monétaire substantielle est compensée par un ré-ajustement de prix à négocier avec l'acheteur final et à formaliser dans un Supplementary Agreement sous 60 jours citeturn20file0.  
            • Enfin, le financement couvert par l'agence COFACE (art 5.3 Financing Clause) assure au fournisseur des encaissements 100 % en euros et le protège contre un défaut de paiement en roubles citeturn20file9.  
            En pratique, le fournisseur n'est exposé qu'à un risque résiduel, plafonné composant par composant par l'Article 13 (10 % max.) et globalement par l'Article 16 (15 % prix contrat)."""
    },
    {
        "question": "A partir du contrat A, peux-tu dresser la liste des actions à mener par le fournisseur en termes de documents à fournir au client ?",
        "expected_answer": """Obligations documentaires (exhaustives à l'Annexe CDRL de l'Exhibit 2) :  
            1. Plan Qualité, Procédures de soudage & QCP (Art 8 + Exhibit 5).  
            2. Plans d'ensemble & de fabrication, calculs et nomenclatures.  
            3. Rapports d'essais d'usine, protocoles FAT et certificats de conformité (Art 10).  
            4. Programmes de montage, notices d'installation, d'exploitation et de maintenance.  
            5. Dossiers environnement & sûreté (Art 9).  
            6. « Incoming Control Certificate » remis après inspection sur site (clau. 4.4).  
            7. Bank Guarantees : APBG et PBG par unité (clau. 5.2.7) citeturn20file10.  
            8. End-User Certificate (Exhibit 10) et Cost Calculation Form dans les 30 j (clau. 5.1.2) citeturn20file1.  
            9. Toutes les mises à jour périodiques requises par le « Contract Schedule » (Exhibit 3) et les rapports mensuels d'avancement.  
            Tout retard sur un document critique déclenche la pénalité de 500 €/semaine prévue à l'art 13.2."""
    },
    {
        "question": "Quelles obligations du contrat A doivent être impérativement intégrées aux contrats qu'ALSTOM signera avec ses fournisseurs ou sous-traitants ?",
        "expected_answer": """Clauses à « flow-down » obligatoires :  
            • Qualité & documentation : Article 8 + Exhibit 5 (respect du QMS, formats de livrables).  
            • Santé-Sécurité-Environnement : Article 9.  
            • Inspections / essais & droit de visite client (Article 10, notamment 10.5-10.9).  
            • Garantie et réparations (Article 12 : 2 ans + extensions après toute réparation).  
            • Pénalités de délai et de performance (Article 13) - mêmes taux pour les sous-traitants que ceux supportés par ALSTOM.  
            • Plafond de responsabilité et exclusions (Article 16) de façon à ce que le cap amont (15 %) reste effectif.  
            • Conformité export / contrôle des données (Article 27).  
            Sans transposition stricte de ces clauses, ALSTOM resterait exposé aux sanctions client qu'elle ne pourrait répercuter."""
    }
]

@dataclass(frozen=True)
class Config:
    model: str
    rerank_model: Optional[str]
    top_k: int
    temperature: float
    similarity_threshold: float
    use_summarize: bool
    chat_mode: str

class RAGBenchmark:
    def __init__(self, test_cases: List[Dict[str, Any]]):
        self.test_cases = test_cases
        self.configs = [Config(m, r, k, t, s, u, c)
                        for m, r, k, t, s, u, c in product(
                            ["mistral-small3.1:latest", "command-a:latest", "qwen3:30b-a3b"],
                            ["bge-reranker-large"],
                            [3, 5, 7, 10],
                            [0.3, 0.5],
                            [0.6, 0.7, 0.8],
                            [True],
                            ["chat"]
                        )]
        self.workspace_dir = Path(__file__).resolve().parent.parent
        self.contract_dir = self.workspace_dir / ".." / "data" / "Contract"
        
        # Modification - Utiliser le chemin absolu de la DB à la racine du projet
        project_root = self.workspace_dir.parent  # Remonte d'un niveau pour atteindre le répertoire racine
        self.db_dir = project_root / "chroma_db"
        logger.info(f"Utilisation de la base de données à: {self.db_dir}")
        
        self._verify_paths()

        # Initialize caches and components
        self.text_cache: Dict[Path, str] = {}
        self.chunk_cache: Dict[Path, Any] = {}
        self.text_vectorizer = TextVectorizer()
        self.chunker = TextChunker()
        self.summarizer = ChunkSummarizer()
        self.rerankers: Dict[str, Reranker] = {}
        # Cache keyed by summarization flag only
        self.db_cache: Dict[bool, Tuple[VectorDBInterface, Any]] = {}

        self.results: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
        self.analysis_event = threading.Event()

        self._prepare_result_files()
        self._init_rerankers()
        self._connect_to_database()

        # Start periodic analysis thread
        self.analysis_thread = threading.Thread(target=self._periodic_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def _verify_paths(self):
        if not self.contract_dir.exists():
            raise FileNotFoundError(f"Contract directory not found: {self.contract_dir}")
        files = [f for f in self.contract_dir.iterdir() if f.suffix.lower() in {'.pdf', '.doc', '.docx'}]
        if not files:
            raise FileNotFoundError("No supported files found in contract directory")
        logger.info(f"Found {len(files)} contract files.")
        
        if not self.db_dir.exists():
            logger.warning(f"Database directory not found: {self.db_dir}. Running in read-only mode with existing database.")

    def _prepare_result_files(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.workspace_dir / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        self.json_file = self.results_dir / f"results_{timestamp}.json"
        self.analysis_file = self.results_dir / f"analysis_{timestamp}.md"
        self.excel_file = self.results_dir / f"detailed_{timestamp}.xlsx"
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"# RAG Benchmark Analysis\nDate: {datetime.now()}\n")

    def _init_rerankers(self):
        for cfg in self.configs:
            key = cfg.rerank_model
            if key and key not in self.rerankers:
                try:
                    self.rerankers[key] = Reranker(key)
                    logger.info(f"Loaded reranker: {key}")
                except Exception as e:
                    logger.warning(f"Skipping reranker {key}: {e}")

    def _extract_text(self, file_path: Path) -> str:
        if file_path not in self.text_cache:
            suffix = file_path.suffix.lower()
            if suffix == '.pdf':
                text, _ = extract_pdf_text(file_path)
            else:
                from document_processing.doc_reader import extract_doc_text
                text, _ = extract_doc_text(file_path)
            self.text_cache[file_path] = text
        return self.text_cache[file_path]

    def _get_chunks(self, file_path: Path) -> Any:
        if file_path not in self.chunk_cache:
            text = self._extract_text(file_path)
            chunks = self.chunker.chunk_text(text, str(file_path))
            self.chunk_cache[file_path] = chunks
        return self.chunk_cache[file_path]

    def _connect_to_database(self):
        """Connect to existing database in read-only mode."""
        logger.info(f"Connecting to database at: {self.db_dir}")
        
        # Check if the directory exists
        if not self.db_dir.exists():
            logger.error(f"Database directory does not exist: {self.db_dir}")
            raise RuntimeError(f"Database directory does not exist: {self.db_dir}")
        
        # Check directory contents
        db_files = list(self.db_dir.glob("*"))
        logger.info(f"Database directory contents: {[f.name for f in db_files]}")
        
        # Create ChromaDB client
        import chromadb
        try:
            # Create client
            chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
            
            # In v0.6.0, list_collections returns string names directly
            collection_names = chroma_client.list_collections()
            logger.info(f"Found collections: {collection_names}")
            
            if not collection_names:
                logger.error("No collections found in database")
                raise RuntimeError("No collections found in database")
            
            # Get first collection
            collection_name = collection_names[0]
            logger.info(f"Using collection: {collection_name}")
            
            # Check if collection has documents
            try:
                collection = chroma_client.get_collection(collection_name)
                count = collection.count()
                logger.info(f"Collection '{collection_name}' contains {count} documents")
            except Exception as e:
                logger.error(f"Error accessing collection: {e}")
                raise
            
            # Create interface
            class ChromaDBInterface:
                def __init__(self, client, collection_name, embeddings_manager):
                    self.client = client
                    self.collection_name = collection_name
                    self.collection = client.get_collection(collection_name)
                    self.embeddings_manager = embeddings_manager
                
                def search(self, query, n_results=5, filter_metadata=None):
                    query_embedding = self.embeddings_manager.get_embeddings([query])[0]
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results * 2,
                        where=filter_metadata,
                    )
                    
                    formatted_results = []
                    seen_originals = set()
                    
                    for i in range(len(results["ids"][0])):
                        metadata = results["metadatas"][0][i]
                        is_summary = metadata.get("is_summary", "false").lower() == "true"
                        original_content = metadata.get("original_content", "")
                        
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
                        elif results["documents"][0][i] not in seen_originals:
                            result = {
                                "id": results["ids"][0][i],
                                "document": results["documents"][0][i],
                                "metadata": metadata,
                                "distance": results["distances"][0][i],
                                "is_summary": False
                            }
                            formatted_results.append(result)
                            
                    formatted_results.sort(key=lambda x: x["distance"])
                    return formatted_results[:n_results]
            
            # Create database interface
            db = ChromaDBInterface(chroma_client, collection_name, self.text_vectorizer)
            
            # Create or load graph
            try:
                from core.interaction import load_or_build_graph
                logger.info("Loading graph with load_or_build_graph")
                graph = load_or_build_graph(db, self.text_vectorizer)
                if not graph:
                    logger.warning("Graph is empty, creating minimal graph")
                    from core.graph_manager import GraphManager
                    graph = GraphManager()
            except Exception as e:
                logger.warning(f"Could not load graph: {e}, creating minimal graph")
                from core.graph_manager import GraphManager
                graph = GraphManager()
            
            # Cache for all configurations
            for use_sum in {cfg.use_summarize for cfg in self.configs}:
                self.db_cache[use_sum] = (db, graph)
                
            logger.info(f"Successfully connected to database and initialized graph")
            
        except Exception as e:
            import traceback
            logger.error(f"Database connection error: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Database connection error: {e}")

    def _run_single(self, cfg: Config, case: Dict[str, Any]) -> None:
        res = {'config': asdict(cfg), 'question': case['question'], 'response_time': None, 'accuracy': None, 'semantic_sim': None}
        try:
            db, graph = self.db_cache[cfg.use_summarize]
            results = db.search(query=case['question'], n_results=cfg.top_k)
            filtered = [d for d in results if d['distance'] <= 1-cfg.similarity_threshold]
            if cfg.chat_mode=='graph':
                aug = get_graph_augmented_results(graph, filtered)
                docs = merge_results(filtered, aug)
            else:
                docs = filtered
                
            # S'assurer que docs contient des chaînes de caractères et non des dictionnaires
            context = []
            for d in docs:
                if isinstance(d, dict) and 'document' in d:
                    context.append(d['document'])
                elif isinstance(d, str):
                    context.append(d)
                else:
                    logger.warning(f"Format inattendu dans les résultats: {type(d)}")

            if cfg.rerank_model and cfg.rerank_model in self.rerankers:
                reranked_docs = self.rerankers[cfg.rerank_model].rerank(case['question'], context, cfg.top_k)
                # Extraire le texte des documents reclassés
                context = []
                for doc in reranked_docs:
                    if isinstance(doc, dict):
                        if 'text' in doc:
                            context.append(doc['text'])
                        elif 'document' in doc:
                            context.append(doc['document'])
                        else:
                            logger.warning(f"Document reclassé sans champ texte reconnu: {doc.keys() if hasattr(doc, 'keys') else type(doc)}")
                    else:
                        context.append(doc)
                
            llm = LLMChat(model=cfg.model)
            prompt = self._build_prompt(context, case['question'])
            t0 = time.time()
            ans = llm.generate(prompt, temperature=cfg.temperature)
            t1 = time.time()
            metrics = self._calculate_metrics(ans, case['expected_answer'])
            res.update({'answer': ans, 'response_time': t1-t0, **metrics})
        except Exception as e:
            logger.exception("Error in run_single")
            res['error'] = str(e)
        with self.results_lock:
            self.results.append(res)
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)

    def _build_prompt(self, context: List[str], question: str) -> str:
        # Vérifier que tous les éléments du contexte sont bien des chaînes
        context_strings = []
        for item in context:
            if isinstance(item, dict) and 'document' in item:
                context_strings.append(item['document'])
            elif isinstance(item, str):
                context_strings.append(item)
            else:
                logger.warning(f"Élément de contexte ignoré car format non reconnu: {type(item)}")
                
        ctx = '\n\n---\n\n'.join(context_strings) if context_strings else 'Aucun contexte pertinent trouvé.'
        return f"Tu es un assistant spécialisé dans l'analyse de contrats.\nContexte:\n{ctx}\nQuestion: {question}\nRéponse:"

    def _calculate_metrics(self, answer: str, expected: str) -> Dict[str, float]:
        # keyword accuracy
        terms = set(expected.lower().split())
        found = sum(1 for t in terms if t in answer.lower())
        acc = found/len(terms) if terms else 0.0
        # semantic similarity using cached embeddings
        emb_ans = self.text_vectorizer.get_embeddings([answer])[0]
        emb_exp = self.text_vectorizer.get_embeddings([expected])[0]
        sim = float(np.dot(emb_ans, emb_exp)/(np.linalg.norm(emb_ans)*np.linalg.norm(emb_exp))) if np.linalg.norm(emb_ans) and np.linalg.norm(emb_exp) else 0.0
        return {'accuracy': acc, 'semantic_sim': sim}

    def _periodic_analysis(self):
        while not self.analysis_event.wait(timeout=60):
            self._write_analysis()
        # un dernier dump à la fin
        self._write_analysis()

    def _write_analysis(self):
        # Use raw DataFrame to preserve 'config' column
        df = pd.DataFrame(self.results)
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis Update: {datetime.now()}\n")
            f.write(f"Total runs: {len(self.results)}\n")
            # Fastest run
            if 'response_time' in df.columns and not df['response_time'].isna().all():
                idx = df['response_time'].idxmin()
                best = df.iloc[idx]
                f.write(f"**Fastest run**: {best['config']} in {best['response_time']:.2f}s\n")
            # Best semantic similarity
            if 'semantic_sim' in df.columns and not df['semantic_sim'].isna().all():
                idx2 = df['semantic_sim'].idxmax()
                bests = df.iloc[idx2]
                f.write(f"**Best semantic sim**: {bests['config']} = {bests['semantic_sim']:.3f}\n")
        with pd.ExcelWriter(self.excel_file) as writer:
            df.to_excel(writer, index=False)

    def run(self):
        print("🚀 Starting RAG benchmark...")
        print(self.configs)
        tasks = [(cfg, case) for cfg in self.configs for case in self.test_cases]
        # On soumet tout et on garde la correspondance future→(cfg,case)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_task = {
                executor.submit(self._run_single, cfg, case): (cfg, case)
                for cfg, case in tasks
            }
            # on crée une barre tqdm
            with tqdm(total=len(future_to_task), desc="Benchmarking") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    cfg, case = future_to_task[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Erreur sur {cfg}, question «{case['question']}» : {e}")
                    # On met à jour la barre avec la combo terminée
                    pbar.set_postfix({
                        "model": cfg.model,
                        "top_k": cfg.top_k,
                        "question": case["question"][:30] + "…"
                    })
                    pbar.update()
        # on arrête proprement le thread périodique
        self.analysis_event.set()
        self.analysis_thread.join()
        print("✅ Benchmark complete.")

if __name__ == '__main__':
    cleanup_flag_documents()
    benchmark = RAGBenchmark(TEST_CASES)
    benchmark.run()
