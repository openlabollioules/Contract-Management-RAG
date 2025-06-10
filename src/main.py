import sys
import platform
import torch
from core.cli_handler import parse_arguments, print_usage, process_arguments
from core.document_manager import clean_document_in_database
from utils.logger import setup_logger

# Configurer le logger pour ce module
logger = setup_logger(__file__)

def info_device():
    is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
    if is_apple_silicon:
        logger.info("🍎 Détection d'un processeur Apple Silicon")
        if torch.backends.mps.is_available():
            logger.info("🎮 GPU MPS disponible")
            device = torch.device("mps")
        else:
            logger.warning("⚠️ GPU MPS non disponible, utilisation du CPU")
            device = torch.device("cpu")
    else:
        logger.info("💻 Architecture non Apple Silicon")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"⚙️ Utilisation du device: {device}")

def main():
    info_device()
    clean_document_in_database()
    args = parse_arguments()
    process_arguments(args)
    exit()

if __name__ == "__main__":
    # Vérifier les arguments basiques
    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_usage()
        sys.exit(0)

    main()
