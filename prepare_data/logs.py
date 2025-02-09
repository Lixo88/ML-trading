import logging
import pandas as pd
import numpy as np
from datetime import datetime
from colorama import Fore, Style, init
from logging.handlers import RotatingFileHandler

# Inicializar colorama para colores en la consola
init(autoreset=True)

# Configurar logger global
logger = logging.getLogger("trading_logs")
logger.setLevel(logging.DEBUG)

def configure_logging(log_file="preparation_log.txt", max_size=5 * 1024 * 1024, backup_count=3):
    """
    Configura el sistema de logging con rotación de archivos y formato mejorado.
    """
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_event(level, message, module="global"):
    """
    Registra un evento en el sistema de logs con colores en la consola y formato mejorado en archivo.
    """
    color_map = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'DEBUG': Fore.CYAN
    }
    color = color_map.get(level.upper(), Fore.WHITE)
    colored_message = f"{color}[{level}] [{module}] {message}{Style.RESET_ALL}"
    plain_message = f"[{level}] [{module}] {message}"  # Sin colores para el archivo
    
    print(colored_message)
    
    logger = logging.getLogger(module)
    if level.upper() == 'INFO':
        logger.info(plain_message)
    elif level.upper() == 'WARNING':
        logger.warning(plain_message)
    elif level.upper() == 'ERROR':
        logger.error(plain_message)
    else:
        logger.debug(plain_message)

# Llamar a configure_logging() una sola vez al inicio del script
configure_logging()
