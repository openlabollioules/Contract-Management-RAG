import sys

from core.contract_processor import process_contract
from core.interaction import chat_with_contract, search_contracts
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.info("Usage: python main.py <contract_file> [search_query|--chat]")
        sys.exit(1)

    filepath = sys.argv[1]

    # If --chat is provided, enter chat mode
    if len(sys.argv) > 2 and sys.argv[2] == "--chat":
        logger.info("\n💬 Mode chat activé. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nVotre question : ")
            if query.lower() == "exit":
                break
            chat_with_contract(query)
    # If search query is provided, perform search
    elif len(sys.argv) > 2:
        search_query = " ".join(sys.argv[2:])
        search_contracts(search_query)
    else:
        # Process the contract
        chunks = process_contract(filepath)
