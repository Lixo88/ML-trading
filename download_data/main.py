import os
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Importar funciones del paquete
from download_data.download_binance_data import (
    load_config, get_binance_client, fetch_with_retries, save_raw_data
)
from download_data.validation import validate_and_correct_data

def setup_logger(log_filename):
    """
    Configura un logger con nombre 'download_data' y lo asocia
    a un FileHandler y un StreamHandler.
    """
    logger = logging.getLogger("download_data")
    logger.setLevel(logging.INFO)

    # Evitar duplicados si se llama varias veces
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Handler para archivo
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Handler para consola
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
    logger = logging.getLogger("download_data.process")
    try:
        klines = fetch_with_retries(client, symbol, interval, start_date, end_date)
        raw_path = save_raw_data(klines, symbol, interval, start_date, end_date, base_output_dir)

        # Validar y corregir
        validated_path = validate_and_correct_data(raw_path, base_output_dir)

        logger.info(f"Proceso finalizado para {symbol} {interval}. Validado en: {validated_path}")

    except Exception as e:
        logger.error(f"Error en process_symbol_timeframe({symbol}, {interval}): {e}", exc_info=True)
        # Relanzar la excepción para que future.result() la capte
        raise

def main():
    # 1) Cargar configuración
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # 2) Preparar rutas para logs y salida
    log_dir = os.path.expanduser(config["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "download_data.log")

    # 3) Configurar logger
    logger = setup_logger(log_filename)
    logger.info("Iniciando proceso de descarga y validación...")

    # 4) Parámetros de configuración
    api_key = config["api_key"]
    api_secret = config["api_secret"]
    symbols = config["symbols"]       # ["BTCUSDT", "ETHUSDT", ...]
    timeframes = config["timeframes"] # ["1d","4h","1h","15m", ...]
    start_date_str = config["start_date"]
    end_date_str = config.get("end_date")

    # Validar fechas
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = None
    if end_date_str:
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        if start_dt > end_dt:
            raise ValueError("start_date no puede ser mayor a end_date")

    # Directorio base
    base_output_dir = os.path.expanduser(config["output_dir"])
    os.makedirs(base_output_dir, exist_ok=True)

    # 5) Crear cliente de Binance
    client = get_binance_client(api_key, api_secret)

    # 6) Descargar + validar en paralelo
    futures = []
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for symbol in symbols:
            for tf in timeframes:
                futures.append(
                    executor.submit(
                        process_symbol_timeframe,
                        client, symbol, tf, start_date_str, end_date_str, base_output_dir
                    )
                )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failed += 1

    logger.info(f"Proceso de descarga y validación completado. Errores: {failed}")

if __name__ == "__main__":
    main()

