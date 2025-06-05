import logging
import sys
from typing import Any, Dict, List
from pathlib import Path

from core.contract_processor import process_contract
from core.document_manager import (delete_document, document_exists,
                                   get_existing_documents)
from core.interaction import chat_with_contract, chat_with_contract_decomposed, chat_with_contract_alternatives, search_contracts, process_query
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
        "Usage: python main.py <contract_file1> [contract_file2 ...] [--chat|--search <search_query>] [--force] [--delete]"
    )
    print("Options:")
    print("  --chat                 Mode chat interactif avec les contrats")
    print("  --advanced-chat        Mode chat avec décomposition des requêtes complexes")
    print("  --alternatives-chat    Mode chat utilisant des requêtes alternatives")
    print("  --graph-chat           Mode chat utilisant le graphe de connaissances")
    print("  --search <query>       Recherche dans les contrats")
    print(
        "  --force                Force le retraitement des documents même s'ils existent déjà"
    )
    print(
        "  --delete               Supprime les documents spécifiés de la base de données"
    )
    print("  --debug                Active le mode debug avec logs détaillés")
    print("  --summarize-chunks     Résume chaque chunk avant de l'ajouter à la base de données")
    print("  --hybrid               Active la recherche hybride (BM25 + sémantique)")
    print("  --no-hybrid            Désactive la recherche hybride (utilise uniquement la recherche sémantique)")
    print("Examples:")
    print("  Process one contract:           python main.py contract.pdf")
    print(
        "  Process multiple contracts:     python main.py contract1.pdf contract2.pdf"
    )
    print("  Force reprocessing:             python main.py contract1.pdf --force")
    print(
        "  Delete documents:               python main.py contract1.pdf contract2.pdf --delete"
    )
    print(
        "  Chat with processed contracts:  python main.py contract1.pdf contract2.pdf --chat"
    )
    print("  Chat with all contracts:        python main.py --chat")
    print(
        "  Advanced chat:                  python main.py --advanced-chat"
    )
    print(
        "  Alternatives chat:              python main.py --alternatives-chat"
    )
    print(
        "  Search in contracts:            python main.py contract1.pdf contract2.pdf --search payment terms"
    )
    print(
        "  Chat with hybrid search:        python main.py --chat --hybrid"
    )


def parse_arguments() -> Dict[str, Any]:
    """
    Parse les arguments de la ligne de commande

    Returns:
        Un dictionnaire contenant les options et valeurs extraites des arguments
    """
    args = {
        "debug": False,
        "force": False,
        "delete": False,
        "chat": False,
        "advanced_chat": False,
        "alternatives_chat": False,
        "graph-chat": False,
        "search": False,
        "search_query": "",
        "filepaths": [],
        "summarize_chunks": False,
        "classification": False,
        "hybrid": bool(os.getenv("USE_HYBRID", "True").lower() == "true"),  # Par défaut, utiliser la valeur de USE_HYBRID dans l'environnement
    }

    # Copier les arguments sans le nom du script
    argv = sys.argv[1:]

    # Vérifier les arguments spéciaux
    if "--debug" in argv:
        args["debug"] = True
        argv.remove("--debug")

    if "--force" in argv:
        args["force"] = True
        argv.remove("--force")

    if "--delete" in argv:
        args["delete"] = True
        argv.remove("--delete")

    if "--chat" in argv:
        args["chat"] = True
        argv.remove("--chat")

    if "--advanced-chat" in argv:
        args["advanced_chat"] = True
        argv.remove("--advanced-chat")

    if "--alternatives-chat" in argv:
        args["alternatives_chat"] = True
        argv.remove("--alternatives-chat")

    if "--graph-chat" in argv:
        args["graph-chat"] = True
        argv.remove("--graph-chat")

    if "--summarize-chunks" in argv:
        args["summarize_chunks"] = True
        argv.remove("--summarize-chunks")
        
    if "--hybrid" in argv:
        args["hybrid"] = True
        argv.remove("--hybrid")
    
    if "--no-hybrid" in argv:
        args["hybrid"] = False
        argv.remove("--no-hybrid")

    if "--search" in argv and argv.index("--search") + 1 < len(argv):
        args["search"] = True
        search_index = argv.index("--search")
        # Extraire tout ce qui suit --search comme query
        args["search_query"] = " ".join(argv[search_index + 1 :])
        # Ne garder que ce qui précède --search pour les fichiers
        argv = argv[:search_index]

    if "--classification" in argv:
        args["classification"] = True
        argv.remove("--classification")

    # Le reste des arguments sont des chemins de fichiers
    args["filepaths"] = [path for path in argv if not path.startswith("--")]

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

        if document_exists(filepath):
            logger.info(f"\n🗑️ Suppression du document: {filepath}")
            if delete_document(filepath):
                logger.info(f"✅ Document {filepath} supprimé avec succès.")
            else:
                logger.error(f"❌ Échec de la suppression du document {filepath}.")
        else:
            logger.warning(
                f"\n⚠️ Le document {filepath} n'existe pas dans la base de données."
            )


def handle_chat_mode(filepaths: List[str], force_reprocess: bool, classification: bool, use_hybrid: bool) -> None:
    """
    Gère le mode chat interactif

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        classification: Si True, utilise la classification des requêtes
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = get_existing_documents(filepaths, force_reprocess)

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
        
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response = process_query(query, use_graph=False, use_classification=classification, use_hybrid=use_hybrid)


def handle_advanced_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool) -> None:
    """
    Gère le mode chat interactif avancé avec décomposition des requêtes

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = get_existing_documents(filepaths, force_reprocess)

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
    
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response, sources = chat_with_contract_decomposed(query, use_graph=False)


def handle_alternatives_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool) -> None:
    """
    Gère le mode chat interactif avec requêtes alternatives

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = get_existing_documents(filepaths, force_reprocess)

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
    
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response, sources = chat_with_contract_alternatives(query, use_graph=False)


def handle_graph_chat_mode(filepaths: List[str], force_reprocess: bool, use_hybrid: bool) -> None:
    """
    Gère le mode chat interactif avec le graphe de connaissances

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
    """
    # Traiter les documents restants avant d'entrer en mode chat
    if filepaths:
        # Vérifier les documents existants
        existing_docs = get_existing_documents(filepaths, force_reprocess)

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
        
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        response = chat_with_contract(query, use_graph=True, use_hybrid=use_hybrid)
        

def handle_search_mode(
    filepaths: List[str], search_query: str, force_reprocess: bool, use_hybrid: bool
) -> None:
    """
    Gère le mode de recherche dans les documents

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant la recherche
        search_query: Requête de recherche
        force_reprocess: Si True, force le retraitement des documents
        use_hybrid: Si True, utilise la recherche hybride (BM25 + sémantique)
    """
    # Vérifier les documents existants
    existing_docs = get_existing_documents(filepaths, force_reprocess)

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
        search_contracts(search_query, use_hybrid=use_hybrid)
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
    existing_docs = get_existing_documents(filepaths, force_reprocess)

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

    # Configurer le niveau de log si mode debug
    if args["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Mode debug activé")

    # Mode suppression
    if args["delete"]:
        handle_delete_mode(args["filepaths"])
        return
    
    setup_history_database(os.getenv("HISTORY_DB_FILE"))

    # Mode chat
    if args["chat"]:

        if args["classification"]:
            classification = True
        else:
            classification = False

        logger.info(f"Classification: {classification}")
        handle_chat_mode(args["filepaths"], args["force"], classification, args["hybrid"])
        return

    # Mode chat avancé
    if args["advanced_chat"]:
        handle_advanced_chat_mode(args["filepaths"], args["force"], args["hybrid"])
        return

    # Mode alternatives chat
    if args["alternatives_chat"]:
        handle_alternatives_chat_mode(args["filepaths"], args["force"], args["hybrid"])
        return

    # Mode graph-chat
    if args["graph-chat"]:
        handle_graph_chat_mode(args["filepaths"], args["force"], args["hybrid"])
        return

    # Mode recherche
    if args["search"]:
        handle_search_mode(args["filepaths"], args["search_query"], args["force"], args["hybrid"])
        return

    # Mode traitement par défaut
    handle_process_mode(args["filepaths"], args["force"], args["summarize_chunks"])
