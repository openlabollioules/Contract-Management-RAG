import logging
import sys
from typing import Any, Dict, List
from pathlib import Path

from core.contract_processor import process_contract
from core.document_manager import (delete_document, is_document_in_database,
                                   filter_existing_documents)
from core.interaction import chat_with_contract, chat_with_contract_using_query_decomposition, chat_with_contract_using_query_alternatives, display_contract_search_results, query_classifier
from utils.logger import setup_logger
from core.history_func import setup_history_database
from dotenv import load_dotenv
import os
# Configurer le logger pour ce module
logger = setup_logger(__file__)

dotenv_path = Path(__file__).parent.parent.parent / "config.env"
load_dotenv(dotenv_path)


def print_usage() -> None:
    """
    Affiche les instructions d'utilisation du programme
    """
    print(
        "Usage: python main.py <contract_file1> [contract_file2 ...] [--chat|--search <search_query>] [options]"
    )
    print("\nModes d'opération:")
    print("  --chat                 Mode chat interactif avec les contrats")
    print("  --advanced-chat        Mode chat avec décomposition des requêtes complexes")
    print("  --alternatives-chat    Mode chat utilisant des requêtes alternatives")
    print("  --graph-chat           Mode chat utilisant le graphe de connaissances")
    print("  --search <query>       Recherche dans les contrats")
    
    print("\nOptions de traitement:")
    print("  --force                Force le retraitement des documents même s'ils existent déjà")
    print("  --delete               Supprime les documents spécifiés de la base de données")
    print("  --summarize-chunks     Résume chaque chunk avant de l'ajouter à la base de données")
    print("  --classification       Active la classification des requêtes")
    
    print("\nOptions de recherche:")
    print("  --hybrid               Active la recherche hybride (BM25 + sémantique)")
    print("  --no-hybrid            Désactive la recherche hybride (utilise uniquement la recherche sémantique)")
    print("  --top-k <number>       Nombre de résultats à récupérer (défaut: 35)")
    
    print("\nOptions de génération:")
    print("  --max-tokens <number>  Limite de tokens pour le contexte (défaut: 60000)")
    print("  --temperature <float>  Température pour la génération LLM (défaut: 0.3)")
    
    print("\nOptions diverses:")
    print("  --debug                Active le mode debug avec logs détaillés")
    
    print("\nExemples:")
    print("  Traiter un contrat:              python main.py contract.pdf")
    print("  Traiter plusieurs contrats:      python main.py contract1.pdf contract2.pdf")
    print("  Forcer le retraitement:          python main.py contract1.pdf --force")
    print("  Supprimer des documents:         python main.py contract1.pdf contract2.pdf --delete")
    print("  Chat avec contrats spécifiques:  python main.py contract1.pdf contract2.pdf --chat")
    print("  Chat avec tous les contrats:     python main.py --chat")
    print("  Chat avancé:                     python main.py --advanced-chat")
    print("  Chat avec requêtes alternatives: python main.py --alternatives-chat")
    print("  Recherche dans les contrats:     python main.py contract1.pdf --search payment terms")
    print("  Chat avec recherche hybride:     python main.py --chat --hybrid")
    print("  Chat avec paramètres:            python main.py --chat --temperature 0.7 --top-k 50")

def parse_arguments() -> Dict[str, Any]:
    """
    Parse les arguments de la ligne de commande

    Returns:
        Un dictionnaire contenant les options et valeurs extraites des arguments
    """
    args = {
        # Modes opératoires
        "debug": False,
        "force": False,
        "delete": False,
        
        # Modes d'interaction
        "chat": False,
        "advanced_chat": False,
        "alternatives_chat": False,
        "graph_chat": False,  # Correction du nom avec underscore au lieu de tiret
        "search": False,
        "search_query": "",
        
        # Options de traitement
        "filepaths": [],
        "summarize_chunks": False,
        "classification": False,
        "hybrid": bool(os.getenv("USE_HYBRID", "True").lower() == "true"),  # Par défaut depuis config.env
        
        # Options avancées
        "max_tokens": int(os.getenv("CONTEXT_WINDOW", "60000")),
        "top_k": int(os.getenv("TOP_K", "35")),
        "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    }

    # Copier les arguments sans le nom du script
    argv = sys.argv[1:]
    
    # Liste des flags booléens simples
    boolean_flags = {
        "--debug": "debug",
        "--force": "force",
        "--delete": "delete",
        "--chat": "chat",
        "--advanced-chat": "advanced_chat",
        "--alternatives-chat": "alternatives_chat",
        "--graph-chat": "graph_chat",  # Correction du nom avec underscore
        "--summarize-chunks": "summarize_chunks",
        "--classification": "classification",
        "--hybrid": "hybrid",
    }
    
    # Traiter les flags booléens simples
    for flag, arg_name in boolean_flags.items():
        if flag in argv:
            args[arg_name] = True
            argv.remove(flag)
    
    # Traiter les flags qui désactivent des options
    if "--no-hybrid" in argv:
        args["hybrid"] = False
        argv.remove("--no-hybrid")
    
    # Traiter les flags avec valeurs
    i = 0
    while i < len(argv):
        # Options avec paramètre
        if argv[i] == "--search" and i + 1 < len(argv):
            args["search"] = True
            # Extraire tout ce qui suit --search comme query
            search_terms = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                search_terms.append(argv[j])
                j += 1
            args["search_query"] = " ".join(search_terms)
            # Supprimer les arguments traités
            argv = argv[:i] + argv[j:]
            continue
        
        # Options numériques
        elif argv[i] == "--max-tokens" and i + 1 < len(argv):
            try:
                args["max_tokens"] = int(argv[i + 1])
                argv = argv[:i] + argv[i+2:]
                continue
            except ValueError:
                logger.warning(f"Valeur invalide pour --max-tokens: {argv[i+1]}")
                i += 2
                continue
        
        elif argv[i] == "--top-k" and i + 1 < len(argv):
            try:
                args["top_k"] = int(argv[i + 1])
                argv = argv[:i] + argv[i+2:]
                continue
            except ValueError:
                logger.warning(f"Valeur invalide pour --top-k: {argv[i+1]}")
                i += 2
                continue
        
        elif argv[i] == "--temperature" and i + 1 < len(argv):
            try:
                args["temperature"] = float(argv[i + 1])
                argv = argv[:i] + argv[i+2:]
                continue
            except ValueError:
                logger.warning(f"Valeur invalide pour --temperature: {argv[i+1]}")
                i += 2
                continue
        
        # Si c'est un flag inconnu, ignorer
        elif argv[i].startswith("--"):
            logger.warning(f"Option inconnue ignorée: {argv[i]}")
            argv.pop(i)
            continue
            
        i += 1

    # Le reste des arguments sont des chemins de fichiers
    args["filepaths"] = [path for path in argv if not path.startswith("--")]
    
    # Validation: s'assurer qu'un seul mode est actif
    active_modes = sum([
        args["chat"], 
        args["advanced_chat"], 
        args["alternatives_chat"], 
        args["graph_chat"],
        args["search"],
        args["delete"]
    ])
    
    if active_modes > 1:
        logger.warning("⚠️ Plusieurs modes sont activés simultanément. Seul le premier mode sera utilisé.")
    
    # Log des arguments
    logger.debug(f"Arguments parsés: {args}")

    return args

def handle_delete_mode(filepaths: List[str]) -> None:
    """
    Gère le mode de suppression des documents

    Args:
        filepaths: Liste des chemins de fichiers à supprimer
    """
    logger.info(
        "⚠️ Mode suppression activé - les documents spécifiés seront supprimés de la base de données"
    )

    for filepath in filepaths:
        # Skip any flags (starting with --)
        if filepath.startswith("--"):
            continue

        if is_document_in_database(filepath):
            logger.info(f"\n🗑️ Suppression du document: {filepath}")
            if delete_document(filepath):
                logger.info(f"✅ Document {filepath} supprimé avec succès.")
            else:
                logger.error(f"❌ Échec de la suppression du document {filepath}.")
        else:
            logger.warning(
                f"\n⚠️ Le document {filepath} n'existe pas dans la base de données."
            )


def handle_chat_mode(filepaths: List[str], force_reprocess: bool, classification: bool, use_hybrid: bool, top_k: int, temperature: float, max_tokens: int) -> None:
    """
    Gère le mode chat interactif

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        classification: Si True, utilise la classification des requêtes
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
        top_k: Nombre de résultats à récupérer
        temperature: Température pour la génération LLM
        max_tokens: Limite de tokens pour le contexte
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = filter_existing_documents(filepaths, force_reprocess)

        # Si des documents existent déjà, afficher une erreur et quitter
        if existing_docs:
            logger.error(
                "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
            )
            for doc in existing_docs:
                logger.error(f"   - {doc}")
            logger.error("\nPour forcer le retraitement, utilisez l'option --force")
            logger.error("Pour supprimer ces documents, utilisez l'option --delete")
            sys.exit(1)

        # Sinon, traiter tous les documents (sauf les flags)
        for filepath in filepaths:
            if not filepath.startswith("--"):
                logger.info(f"\n📄 Traitement du contrat: {filepath}")
                process_contract(filepath)

    # Entrer en mode chat interactif
    logger.info("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
    if use_hybrid:
        logger.info("🔍 Recherche hybride (BM25 + sémantique) activée")
    else:
        logger.info("🔍 Recherche sémantique standard activée")
    
    logger.info(f"📊 Paramètres: top_k={top_k}, temperature={temperature}, max_tokens={max_tokens}")
        
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response = query_classifier(
            query, 
            n_context=top_k,
            use_graph=False, 
            use_classification=classification, 
            use_hybrid=use_hybrid,
            temperature=temperature,
            context_window=max_tokens
        )


def handle_advanced_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool, top_k: int, temperature: float, max_tokens: int) -> None:
    """
    Gère le mode chat interactif avancé avec décomposition des requêtes

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
        top_k: Nombre de résultats à récupérer
        temperature: Température pour la génération LLM
        max_tokens: Limite de tokens pour le contexte
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = filter_existing_documents(filepaths, force_reprocess)

        # Si des documents existent déjà, afficher une erreur et quitter
        if existing_docs:
            logger.error(
                "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
            )
            for doc in existing_docs:
                logger.error(f"   - {doc}")
            logger.error("\nPour forcer le retraitement, utilisez l'option --force")
            logger.error("Pour supprimer ces documents, utilisez l'option --delete")
            sys.exit(1)

        # Sinon, traiter tous les documents (sauf les flags)
        for filepath in filepaths:
            if not filepath.startswith("--"):
                logger.info(f"\n📄 Traitement du contrat: {filepath}")
                process_contract(filepath)

    # Entrer en mode chat interactif avancé
    logger.info("\n🧠 Mode chat avancé activé (décomposition des requêtes). Tapez 'exit' pour quitter.")
    print("\n💡 Ce mode décompose automatiquement les questions complexes en sous-questions pour une meilleure précision.")
    
    if use_hybrid:
        logger.info("🔍 Recherche hybride (BM25 + sémantique) activée")
    else:
        logger.info("🔍 Recherche sémantique standard activée")
    
    logger.info(f"📊 Paramètres: top_k={top_k}, temperature={temperature}, max_tokens={max_tokens}")
    
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response, sources = chat_with_contract_using_query_decomposition(
            query, 
            n_context=top_k,
            use_graph=False,
            temperature=temperature,
            context_window=max_tokens
        )


def handle_alternatives_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool, top_k: int, temperature: float, max_tokens: int) -> None:
    """
    Gère le mode chat interactif avec requêtes alternatives

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
        top_k: Nombre de résultats à récupérer
        temperature: Température pour la génération LLM
        max_tokens: Limite de tokens pour le contexte
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = filter_existing_documents(filepaths, force_reprocess)

        # Si des documents existent déjà, afficher une erreur et quitter
        if existing_docs:
            logger.error(
                "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
            )
            for doc in existing_docs:
                logger.error(f"   - {doc}")
            logger.error("\nPour forcer le retraitement, utilisez l'option --force")
            logger.error("Pour supprimer ces documents, utilisez l'option --delete")
            sys.exit(1)

        # Sinon, traiter tous les documents (sauf les flags)
        for filepath in filepaths:
            if not filepath.startswith("--"):
                logger.info(f"\n📄 Traitement du contrat: {filepath}")
                process_contract(filepath)

    # Entrer en mode chat interactif avec requêtes alternatives
    logger.info("\n🔍 Mode chat avec requêtes alternatives activé. Tapez 'exit' pour quitter.")
    print("\n💡 Ce mode génère automatiquement des requêtes alternatives pour améliorer la recherche.")
    
    if use_hybrid:
        logger.info("🔍 Recherche hybride (BM25 + sémantique) activée")
    else:
        logger.info("🔍 Recherche sémantique standard activée")
    
    logger.info(f"📊 Paramètres: top_k={top_k}, temperature={temperature}, max_tokens={max_tokens}")
    
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response, sources = chat_with_contract_using_query_alternatives(
            query,
            n_context=top_k,
            use_graph=False,
            temperature=temperature,
            context_window=max_tokens
        )


def handle_graph_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool, top_k: int, temperature: float, max_tokens: int) -> None:
    """
    Gère le mode chat interactif avec le graphe de connaissances

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
        top_k: Nombre de résultats à récupérer
        temperature: Température pour la génération LLM
        max_tokens: Limite de tokens pour le contexte
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = filter_existing_documents(filepaths, force_reprocess)

        # Si des documents existent déjà, afficher une erreur et quitter
        if existing_docs:
            logger.error(
                "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
            )
            for doc in existing_docs:
                logger.error(f"   - {doc}")
            logger.error("\nPour forcer le retraitement, utilisez l'option --force")
            logger.error("Pour supprimer ces documents, utilisez l'option --delete")
            sys.exit(1)

        # Sinon, traiter tous les documents (sauf les flags)
        for filepath in filepaths:
            if not filepath.startswith("--"):
                logger.info(f"\n📄 Traitement du contrat: {filepath}")
                process_contract(filepath)

    # Entrer en mode chat interactif avec graphe
    logger.info("\n🔍 Mode chat augmenté par graphe de connaissances activé. Tapez 'exit' pour quitter.")
    
    if use_hybrid:
        logger.info("🔍 Recherche hybride (BM25 + sémantique) activée")
    else:
        logger.info("🔍 Recherche sémantique standard activée")
    
    logger.info(f"📊 Paramètres: top_k={top_k}, temperature={temperature}, max_tokens={max_tokens}")
        
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response = chat_with_contract(
            query, 
            use_graph=True, 
            use_hybrid=use_hybrid,
            n_context=top_k,
            temperature=temperature,
            context_window=max_tokens
        )


def handle_search_mode(
    filepaths: List[str], search_query: str, force_reprocess: bool, use_hybrid: bool, top_k: int
) -> None:
    """
    Gère le mode de recherche dans les documents

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant la recherche
        search_query: Requête de recherche
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
        top_k: Nombre de résultats à récupérer
    """
    # Vérifier les documents existants
    existing_docs = filter_existing_documents(filepaths, force_reprocess)

    # Si des documents existent déjà, afficher une erreur et quitter
    if existing_docs:
        logger.error(
            "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
        )
        for doc in existing_docs:
            logger.error(f"   - {doc}")
        logger.error("\nPour forcer le retraitement, utilisez l'option --force")
        logger.error("Pour supprimer ces documents, utilisez l'option --delete")
        sys.exit(1)

    # Sinon, traiter tous les documents (sauf les flags)
    for filepath in filepaths:
        if not filepath.startswith("--"):
            logger.info(f"\n📄 Traitement du contrat: {filepath}")
            process_contract(filepath)

    # Effectuer la recherche
    if search_query:
        logger.info(f"📊 Paramètres: top_k={top_k}, hybrid={use_hybrid}")
        display_contract_search_results(search_query, n_results=top_k, use_hybrid=use_hybrid)
    else:
        logger.info("Erreur: Aucune requête de recherche fournie après --search")
        sys.exit(1)


def handle_process_mode(filepaths: List[str], force_reprocess: bool, summarize_chunks: bool = False) -> None:
    """
    Gère le mode de traitement des documents

    Args:
        filepaths: Liste des chemins de fichiers à traiter
        force_reprocess: Si True, force le retraitement des documents
        summarize_chunks: Si True, résume chaque chunk avec Ollama
    """
    # Vérifier les documents existants
    existing_docs = filter_existing_documents(filepaths, force_reprocess)

    # Si des documents existent déjà, afficher une erreur et quitter
    if existing_docs:
        logger.error(
            "\n❌ ERREUR : Les documents suivants existent déjà dans la base de données :"
        )
        for doc in existing_docs:
            logger.error(f"   - {doc}")
        logger.error("\nPour forcer le retraitement, utilisez l'option --force")
        logger.error("Pour supprimer ces documents, utilisez l'option --delete")
        sys.exit(1)

    # Sinon, traiter tous les documents (sauf les flags)
    for filepath in filepaths:
        if not filepath.startswith("--"):
            logger.info(f"\n📄 Traitement du contrat: {filepath}")
            process_contract(filepath, summarize_chunks=summarize_chunks)


def process_arguments(args: Dict[str, Any]) -> None:
    """
    Traite les arguments selon le mode d'opération

    Args:
        args: Dictionnaire contenant les options et valeurs extraites des arguments
    """
    logger.debug(f"Arguments: {args}")
    
    # Configurer le niveau de log si mode debug
    if args["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Mode debug activé")

    # Mode suppression
    if args["delete"]:
        handle_delete_mode(args["filepaths"])
        return
    
    # Initialisation de la base de données d'historique
    setup_history_database(os.getenv("HISTORY_DB_FILE"))

    # Mode chat
    if args["chat"]:
        classification = args["classification"]
        logger.info(f"Classification: {classification}")
        handle_chat_mode(
            args["filepaths"], 
            args["force"], 
            classification, 
            args["hybrid"],
            args["top_k"],
            args["temperature"],
            args["max_tokens"]
        )
        return

    # Mode chat avancé
    if args["advanced_chat"]:
        handle_advanced_chat_mode(
            args["filepaths"], 
            args["force"], 
            args["hybrid"],
            args["top_k"],
            args["temperature"],
            args["max_tokens"]
        )
        return

    # Mode alternatives chat
    if args["alternatives_chat"]:
        handle_alternatives_chat_mode(
            args["filepaths"], 
            args["force"], 
            args["hybrid"],
            args["top_k"],
            args["temperature"],
            args["max_tokens"]
        )
        return

    # Mode graph_chat
    if args["graph_chat"]:
        handle_graph_chat_mode(
            args["filepaths"], 
            args["force"], 
            args["hybrid"],
            args["top_k"],
            args["temperature"],
            args["max_tokens"]
        )
        return

    # Mode recherche
    if args["search"]:
        handle_search_mode(
            args["filepaths"], 
            args["search_query"], 
            args["force"], 
            args["hybrid"],
            args["top_k"]
        )
        return

    # Mode traitement par défaut
    handle_process_mode(args["filepaths"], args["force"], args["summarize_chunks"])
