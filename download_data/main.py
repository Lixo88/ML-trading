# main.py

import os
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar funciones del paquete
from download_data.download_binance_data import (
    load_config, get_binance_client, fetch_with_retries, save_raw_data
)
from download_data.validation import validate_and_correct_data

def setup_logger(log_filename="download_data.log"):
    """
    Configura un logger unificado para el paquete download_data.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Limpia posibles handlers previos

    # Formato
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Handler para archivo
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Handler para consola (opcional)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def process_symbol_timeframe(client, symbol, interval, start_date, end_date, base_output_dir):
    """
    Proceso completo para un (symbol, interval):
    1) Descarga klines
    2) Guarda en raw_data
    3) Valida y corrige gaps
    """
    try:
        klines = fetch_with_retries(client, symbol, interval, start_date, end_date)
        raw_path = save_raw_data(klines, symbol, interval, start_date, end_date, base_output_dir)

        # Validar y corregir
        validated_path = validate_and_correct_data(raw_path, base_output_dir)

        logging.info(f"Proceso finalizado para {symbol} {interval}. Validado en: {validated_path}")

    except Exception as e:
        logging.error(f"Error en process_symbol_timeframe({symbol}, {interval}): {e}")


def main():
    # 1) Cargar configuración
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # 2) Configurar logger
    logger = setup_logger(log_filename="download_data.log")
    logger.info("Iniciando proceso de descarga y validación...")

    # 3) Parámetros
    api_key = config["api_key"]
    api_secret = config["api_secret"]
    symbols = config["symbols"]       # Lista de símbolos, ej: ["BTCUSDT", "ETHUSDT"]
    timeframes = config["timeframes"] # Lista, ej: ["1d", "4h", "1h"]
    start_date = config["start_date"] # Ej: "2021-01-01"
    end_date = config.get("end_date") # Opcional, puede ser None

    # Directorio base donde se guardan los datos procesados
    base_output_dir = os.path.expanduser("~/data_processed")

    # 4) Crear cliente de Binance
    client = get_binance_client(api_key, api_secret)

    # 5) Descarga + validación en paralelo
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for symbol in symbols:
            for tf in timeframes:
                futures.append(
                    executor.submit(
                        process_symbol_timeframe,
                        client, symbol, tf, start_date, end_date, base_output_dir
                    )
                )

        # Esperar a que todas las tareas terminen
        for future in as_completed(futures):
            future.result()  # si hay excepción, se propagará aquí

    logger.info("Proceso de descarga y validación completado para todos los símbolos.")

if __name__ == "__main__":
    main()
