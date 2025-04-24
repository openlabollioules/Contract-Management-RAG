import sys
import logging

from core.document_manager import cleanup_flag_documents
from core.cli_handler import parse_arguments, process_arguments, print_usage
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)

if __name__ == "__main__":
    # Nettoyer les flags incorrectement stockés
    cleanup_flag_documents()
    
    # Vérifier les arguments basiques
    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_usage()
        sys.exit(0)
    
    # Analyser les arguments
    args = parse_arguments()
    
    # Traiter les arguments selon le mode d'opération
    process_arguments(args)
