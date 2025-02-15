"""
logs.py

Un logger estructurado que:
 - Escribe JSON al archivo de log (nivel INFO, WARNING, ERROR, etc.).
 - Muestra en consola con colores (usando colorama).
 - Provee funciones de alto nivel: log_event(level, msg, module, metadata), log_feature_stats(df, features, module).

Ejemplo de Uso:
 from prepare_data.logs import log_event, log_feature_stats

 log_event("INFO", "Iniciando pipeline...", module="main")
 log_feature_stats(df, ["rsi","macd_line"], module="indicators")
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# colorama para colores en consola
from colorama import init, Fore, Style

init(autoreset=True)

class _ColoredFormatter(logging.Formatter):
    """
    Sobrescribe el formateo para aplicar color en consola 
    según el nivel de logging (INFO, WARNING, ERROR...).
    """
    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.BLUE
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return self.COLORS.get(record.levelname, Fore.WHITE) + message + Style.RESET_ALL

class StructuredLogger:
    """
    Logger centralizado que:
     - Escribe JSON al archivo (con nivel, mensaje, module, metadata).
     - Muestra colores en consola (para nivel INFO, WARNING, ERROR...).
    """
    def __init__(self):
        # Nombre del logger
        self.logger = logging.getLogger("prepare_data")
        self.logger.setLevel(logging.INFO)

        # Ruta de logs
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Nombre del archivo => e.g. pipeline_YYYYMMDD.log
        log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

        # Formato base (sin color) para el archivo
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # 1) File handler (JSON en texto)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 2) Console handler (con color)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_ColoredFormatter('%(asctime)s | %(levelname)s | %(message)s'))

        # Agregar al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_event(self, level: str, message: str, module: str = "", metadata: Dict[str, Any] = None):
        """
        Logea un evento con estructura JSON en el archivo,
        y colores en la consola.

        Args:
          level: 'INFO','WARNING','ERROR','DEBUG', etc.
          message: Texto principal.
          module: Nombre del submódulo (indicators, zones, etc.).
          metadata: Diccionario con info extra (opcional).
        """
        if metadata is None:
            metadata = {}

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "module": module,
            "message": message,
            "metadata": metadata
        }

        log_line = json.dumps(log_entry)
        level_up = level.upper()

        if level_up == "INFO":
            self.logger.info(log_line)
        elif level_up == "WARNING":
            self.logger.warning(log_line)
        elif level_up == "ERROR":
            self.logger.error(log_line)
        elif level_up == "DEBUG":
            self.logger.debug(log_line)
        else:
            self.logger.info(log_line)

    def log_feature_stats(self, df: pd.DataFrame, features: List[str], module: str):
        """
        Logea estadísticas básicas (mean, std, quantiles, NaNs) de una lista de features.
        """
        stats = {}
        for feat in features:
            if feat in df.columns:
                series = df[feat].dropna()
                stats[feat] = {
                    "count": int(series.count()),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "q25": series.quantile(0.25),
                    "median": series.quantile(0.5),
                    "q75": series.quantile(0.75),
                    "max": series.max(),
                    "nan_count": int(df[feat].isna().sum())
                }
        self.log_event("INFO", "Feature stats", module, metadata={"stats": stats})

# Instancia global para uso directo
_logger = StructuredLogger()

def log_event(level: str, msg: str, module: str = "", metadata: Dict[str, Any] = None):
    """
    Función global para que el pipeline llame: 
    from prepare_data.logs import log_event
    log_event("INFO","Arrancando", module="main")
    """
    _logger.log_event(level, msg, module, metadata)

def log_feature_stats(df: pd.DataFrame, features: List[str], module: str):
    """
    Llama a la versión interna del logger.
    from prepare_data.logs import log_feature_stats
    log_feature_stats(df, ["rsi","macd_line"], "indicators")
    """
    _logger.log_feature_stats(df, features, module)
