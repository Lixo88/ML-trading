# prepare_data/logs.py

import logging
import os
from colorama import init, Fore, Style

# Inicializa colorama (para soportar ANSI en Windows, etc.)
init(autoreset=True)

# Directorio de logs (subiendo un nivel desde el actual __file__ y creando "logs")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Nombre del archivo de log
LOG_FILENAME = os.path.join(LOGS_DIR, "prepare_data.log")

# Creamos un logger con nombre "prepare_data"
logger = logging.getLogger("prepare_data")
logger.setLevel(logging.INFO)

# Formato básico (sin colores) que se escribirá tanto en archivo como consola
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Handler para archivo
file_handler = logging.FileHandler(LOG_FILENAME, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------------------------
# Handler de consola con colores
# -----------------------------------------------
class ColorConsoleHandler(logging.StreamHandler):
    """
    Sobrescribe emit para aplicar color según el nivel de log (INFO, WARNING, ERROR, etc.)
    """
    def emit(self, record):
        # Copiamos el mensaje original
        original_msg = record.msg

        # Definimos un color por default
        color = Fore.WHITE

        if record.levelno == logging.INFO:
            color = Fore.GREEN
        elif record.levelno == logging.WARNING:
            color = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        elif record.levelno == logging.DEBUG:
            color = Fore.BLUE

        # Agregamos color ANSI al mensaje
        record.msg = color + str(record.msg) + Style.RESET_ALL

        # Llamamos a la implementación base para emitir
        super().emit(record)

        # Restauramos el mensaje original (para que no se quede con ANSI en el file)
        record.msg = original_msg

# Creamos el handler de consola y lo agregamos al logger
console_handler = ColorConsoleHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_event(level: str, msg: str, module: str = ""):
    """
    Función principal para logear eventos. 
    - level: 'INFO', 'WARNING', 'ERROR', 'DEBUG', etc.
    - msg: texto del mensaje
    - module: el submódulo ('indicators', 'zones', etc.)

    Muestra color en consola y graba en el archivo prepare_data.log sin códigos ANSI.
    """
    level_up = level.upper()
    # Si se especifica el módulo, agregamos "[modulo]" al inicio del msg.
    final_msg = f"[{module}] {msg}" if module else msg

    if level_up == "INFO":
        logger.info(final_msg)
    elif level_up == "WARNING":
        logger.warning(final_msg)
    elif level_up == "ERROR":
        logger.error(final_msg)
    elif level_up == "DEBUG":
        logger.debug(final_msg)
    else:
        # Default a INFO si no reconocemos el nivel
        logger.info(final_msg)
