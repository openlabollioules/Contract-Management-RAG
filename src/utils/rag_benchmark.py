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
                            ["command-a:latest"],
                            [None, "bge-reranker-large", "Jina-ColBERT-v1"],
                            [3, 5, 7, 10],
                            [0.1, 0.3, 0.5, 0.7],
                            [0.5, 0.6, 0.7, 0.8],
                            [True],
                            ["graph"]
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
        """Connect to existing database in read-only mode without resetting or modifying it."""
        logger.info(f"Connecting to existing database at: {self.db_dir}")
        
        # Création d'une seule instance de ChromaDB
        import chromadb
        try:
            # Créer une seule instance du client ChromaDB
            chroma_client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,  # Mode lecture seule
                    is_persistent=True
                )
            )
            
            # Lister les collections disponibles - ChromaDB 0.6.0 retourne directement les noms
            try:
                collection_names = chroma_client.list_collections()
                logger.info(f"Collections existantes dans la base: {collection_names}")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des collections: {e}")
                # Essai direct avec la collection "contracts"
                collection_names = ["contracts"]
            
            # Adapter notre classe VectorDBInterface pour utiliser le client existant
            class ReadOnlyVectorDBInterface:
                def __init__(self, client, collection_name, text_vectorizer):
                    self.collection = client.get_collection(collection_name)
                    self.embeddings_manager = text_vectorizer
                
                def search(self, query, n_results=5, filter_metadata=None):
                    logger.info(f"Recherche dans ChromaDB: '{query}' (n_results={n_results})")
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
            
            # Si aucune collection n'est trouvée, essayons d'utiliser collection_names[0]
            if not collection_names:
                raise RuntimeError("Aucune collection n'existe dans la base de données")
            
            # Pour chaque configuration, utiliser la première collection disponible
            for use_sum in {cfg.use_summarize for cfg in self.configs}:
                if use_sum in self.db_cache:
                    continue
                
                try:
                    # Utiliser la première collection disponible
                    collection_name = collection_names[0]
                    logger.info(f"Utilisation de la collection: {collection_name}")
                    
                    # Créer notre wrapper de vectordb avec le client existant
                    db = ReadOnlyVectorDBInterface(chroma_client, collection_name, self.text_vectorizer)
                    
                    # Valider que la collection contient des documents
                    results = db.search("test", n_results=1)
                    if not results:
                        logger.warning(f"Collection '{collection_name}' est vide")
                        continue
                    
                    # Tenter de charger le graphe depuis le fichier
                    graph_path = self.workspace_dir.parent / "knowledge_graph.pkl"
                    if graph_path.exists():
                        try:
                            # Charger directement avec pickle
                            import pickle
                            logger.info(f"Chargement du graphe depuis: {graph_path}")
                            with open(str(graph_path), 'rb') as f:
                                graph = pickle.load(f)
                                if graph:
                                    logger.info(f"Graphe chargé avec succès depuis: {graph_path}")
                                    self.db_cache[use_sum] = (db, graph)
                                    logger.info(f"Base de données et graphe chargés avec succès pour use_summarize={use_sum}")
                                    continue
                        except Exception as e:
                            logger.warning(f"Échec du chargement du graphe: {e}")
                    
                    # Si nous sommes arrivés ici, c'est que nous n'avons pas pu charger le graphe
                    # On va charger le graphe plus simplement avec la fonction existante
                    try:
                        from core.interaction import load_or_build_graph
                        logger.info("Utilisation de load_or_build_graph")
                        graph = load_or_build_graph(db, self.text_vectorizer)
                        if graph:
                            self.db_cache[use_sum] = (db, graph)
                            logger.info(f"Graphe chargé/construit avec succès pour use_summarize={use_sum}")
                            continue
                        else:
                            logger.warning("Échec de load_or_build_graph - graphe vide ou nul")
                    except Exception as e:
                        logger.warning(f"Échec de load_or_build_graph: {e}")
                
                except Exception as e:
                    logger.warning(f"Échec avec la collection '{collection_name}': {e}")
            
            # Vérifier si nous avons réussi à configurer au moins une base de données
            if not self.db_cache:
                raise RuntimeError("Impossible de se connecter à une base de données valide")
                
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données: {e}")
            raise RuntimeError(f"Impossible de se connecter à une base de données valide: {e}")

    def _build_prompt(self, context: List[str], question: str) -> str:
        ctx = '\n\n---\n\n'.join(context) if context else 'Aucun contexte pertinent trouvé.'
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
            context = [d['document'] for d in docs]
            if cfg.rerank_model and cfg.rerank_model in self.rerankers:
                context = self.rerankers[cfg.rerank_model].rerank(case['question'], context, cfg.top_k)
            llm = LLMChat(model=cfg.model)
            prompt = self._build_prompt(context, case['question'])
            t0 = time.time()
            ans = llm.generate(prompt, options={'temperature':cfg.temperature})
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
