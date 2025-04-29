import logging
import sys
from typing import Any, Dict, List

from core.contract_processor import process_contract
from core.document_manager import (delete_document, document_exists,
                                   get_existing_documents)
from core.interaction import chat_with_contract, search_contracts
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


def print_usage() -> None:
    """
    Affiche les instructions d'utilisation du programme
    """
    print(
        "Usage: python main.py <contract_file1> [contract_file2 ...] [--chat|--search <search_query>] [--force] [--delete]"
    )
    print("Options:")
    print("  --chat                 Mode chat interactif avec les contrats")
    print("  --search <query>       Recherche dans les contrats")
    print(
        "  --force                Force le retraitement des documents même s'ils existent déjà"
    )
    print(
        "  --delete               Supprime les documents spécifiés de la base de données"
    )
    print("  --debug                Active le mode debug avec logs détaillés")
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
        "  Search in contracts:            python main.py contract1.pdf contract2.pdf --search payment terms"
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
        "graph-chat": False,
        "search": False,
        "search_query": "",
        "filepaths": [],
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

    if "--graph-chat" in argv:
        args["graph-chat"] = True
        argv.remove("--graph-chat")

    if "--search" in argv and argv.index("--search") + 1 < len(argv):
        args["search"] = True
        search_index = argv.index("--search")
        # Extraire tout ce qui suit --search comme query
        args["search_query"] = " ".join(argv[search_index + 1 :])
        # Ne garder que ce qui précède --search pour les fichiers
        argv = argv[:search_index]

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


def handle_chat_mode(filepaths: List[str], force_reprocess: bool) -> None:
    """
    Gère le mode chat interactif

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
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
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        chat_with_contract(query, use_graph=False)

def handle_graph_chat_mode(filepaths: List[str], force_reprocess: bool) -> None:
    """
    Gère le mode chat interactif avec le graphe de connaissances

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant le chat
        force_reprocess: Si True, force le retraitement des documents
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
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break
        chat_with_contract(query, use_graph=True)
        
def handle_search_mode(
    filepaths: List[str], search_query: str, force_reprocess: bool
) -> None:
    """
    Gère le mode de recherche dans les documents

    Args:
        filepaths: Liste des chemins de fichiers à traiter avant la recherche
        search_query: Requête de recherche
        force_reprocess: Si True, force le retraitement des documents
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
        search_contracts(search_query)
    else:
        logger.info("Erreur: Aucune requête de recherche fournie après --search")
        sys.exit(1)


def handle_process_mode(filepaths: List[str], force_reprocess: bool) -> None:
    """
    Gère le mode de traitement des documents (mode par défaut)

    Args:
        filepaths: Liste des chemins de fichiers à traiter
        force_reprocess: Si True, force le retraitement des documents
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


def process_arguments(args: Dict[str, Any]) -> None:
    """
    Traite les arguments analysés selon le mode d'opération

    Args:
        args: Dictionnaire des arguments et options
    """
    # Configurer le mode debug si demandé
    if args["debug"]:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔍 Mode debug activé - affichage des logs détaillés")

    # Gestion du mode de suppression
    if args["delete"]:
        handle_delete_mode(args["filepaths"])
        sys.exit(0)

    # Gestion du mode chat
    elif args["chat"]:
        handle_chat_mode(args["filepaths"], args["force"])
        sys.exit(0)

    elif args["graph-chat"]:
        handle_graph_chat_mode(args["filepaths"], args["force"])
        sys.exit(0)

    # Gestion du mode recherche
    elif args["search"]:
        handle_search_mode(args["filepaths"], args["search_query"], args["force"])
        sys.exit(0)

    # Mode par défaut: traitement des documents
    else:
        handle_process_mode(args["filepaths"], args["force"])
