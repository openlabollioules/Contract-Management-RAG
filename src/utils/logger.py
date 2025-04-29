import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier de config
if os.path.exists("logger_config.env"):
    load_dotenv("logger_config.env")
else:
    # Créer un fichier de configuration par défaut si celui-ci n'existe pas
    Path("./Logs").mkdir(exist_ok=True)
    default_config = """# Les Logger
### Fichier main.py
LOG_FILE_main="./Logs/main.log"
LOG_LEVEL_main="INFO"
"""
    with open("logger_config.env", "w") as f:
        f.write(default_config)
    load_dotenv("logger_config.env")

# Créer le répertoire des logs s'il n'existe pas
Path("./Logs").mkdir(exist_ok=True)


def setup_logger(module_name: str):
    """
    Configure un logger pour un module spécifique avec les paramètres définis dans logger_config.env

    Args:
        module_name: Le nom du module pour lequel configurer le logger

    Returns:
        Un logger configuré
    """
    # Nom de fichier basé sur le module sans l'extension .py
    module_base_name = module_name.split("/")[-1].replace(".py", "")

    # Récupérer les variables depuis logger_config.env ou utiliser des valeurs par défaut
    log_file = os.getenv(
        f"LOG_FILE_{module_base_name}", f"./Logs/{module_base_name}.log"
    )
    log_level_str = os.getenv(f"LOG_LEVEL_{module_base_name}", "INFO")

    # Convertir le niveau de log en constante
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Créer le logger
    logger = logging.getLogger(module_base_name)

    # Prevent log message propagation to the root logger
    logger.propagate = False

    # Remove all handlers if any exist (to avoid duplicates)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Format de log avec timestamp, niveau, nom du fichier, numéro de ligne et fonction
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )

    # Handler pour fichier (avec rotation à 5 Mo et garde 3 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(log_format)

    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
