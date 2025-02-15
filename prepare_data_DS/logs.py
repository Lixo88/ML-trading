# logs.py
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
from colorama import Fore, Style, init
import pandas as pd

init(autoreset=True)  # Para colores en consola

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("ML-Trading")
        self.logger.setLevel(logging.INFO)
        
        # Configurar formato estructurado
        self.formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        # Handler para archivo
        self.file_handler = logging.FileHandler(
            self._get_log_path(),
            encoding='utf-8'
        )
        self.file_handler.setFormatter(self.formatter)
        
        # Handler para consola con colores
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self._ColoredFormatter())
        
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
    
    class _ColoredFormatter(logging.Formatter):
        COLORS = {
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        
        def format(self, record):
            message = super().format(record)
            return self.COLORS.get(record.levelname, Fore.WHITE) + message + Style.RESET_ALL
    
    def _get_log_path(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    
    def log_event(self, level: str, message: str, module: str, metadata: Dict[str, Any] = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "module": module,
            "message": message,
            "metadata": metadata or {}
        }
        
        if level.upper() == 'INFO':
            self.logger.info(json.dumps(log_entry))
        elif level.upper() == 'WARNING':
            self.logger.warning(json.dumps(log_entry))
        elif level.upper() == 'ERROR':
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.debug(json.dumps(log_entry))
    
    def log_feature_stats(self, df: pd.DataFrame, features: list, module: str):
        stats = {}
        for feat in features:
            if feat in df.columns:
                stats[feat] = {
                    "mean": df[feat].mean(),
                    "std": df[feat].std(),
                    "q25": df[feat].quantile(0.25),
                    "q75": df[feat].quantile(0.75),
                    "nan_count": df[feat].isna().sum()
                }
        self.log_event('INFO', "Feature statistics", module, {"stats": stats})

# Instancia global
logger = StructuredLogger()