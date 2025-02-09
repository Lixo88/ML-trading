import os
import time
from typing import Union
import pandas as pd
import logging
from binance.client import Client
from datetime import datetime
import yaml

def load_config(config_file: str) -> dict:
    """
    Carga la configuración desde un archivo YAML.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_binance_client(api_key: str, api_secret: str) -> Client:
    """
    Crea y retorna el cliente de la API de Binance.
    """
    logger = logging.getLogger("download_data.binance")
    logger.info("Creando cliente de Binance...")
    return Client(api_key, api_secret)

def fetch_with_retries(
    client: Client,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str = None,
    max_retries: int = 5
):
    """
    Descarga velas de Binance con paginación y reintentos exponenciales.
    Retorna una lista de klines (raw).
    """
    logger = logging.getLogger("download_data.binance")
    logger.info(f"Iniciando descarga para {symbol} - {interval}. "
                f"Rango: {start_date} a {end_date if end_date else 'latest'}")

    all_klines = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = None
    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    while True:
        for attempt in range(max_retries):
            try:
                klines = client.get_historical_klines(
                    symbol, interval, start_ts, end_str=end_ts, limit=1000
                )
                time.sleep(0.2)  # Pequeño delay para evitar rate limits

                if not klines:
                    logger.info(f"No se encontraron más datos para {symbol} {interval}.")
                    return all_klines

                # Agregamos un chequeo de estructura (solo si hay datos):
                if len(klines[0]) != 12:
                    raise ValueError("Estructura inesperada de klines (se esperan 12 columnas).")

                all_klines.extend(klines)
                last_close_time = klines[-1][6]
                start_ts = last_close_time + 1

                if end_ts and last_close_time >= end_ts:
                    logger.info(f"Descarga finalizada para {symbol} {interval}. "
                                f"Total filas: {len(all_klines)}")
                    return all_klines

                # Si se descargaron menos de 1000 velas, hemos llegado al final
                if len(klines) < 1000:
                    logger.info(f"Descarga completa para {symbol} {interval}. "
                                f"Total filas: {len(all_klines)}")
                    return all_klines

                break  # salir del for de reintentos si fue exitoso
            except Exception as e:
                logger.warning(f"Error en descarga de {symbol} {interval}: {e}. "
                               f"Reintento {attempt+1}/{max_retries}...")
                time.sleep(2 ** attempt)  # espera exponencial

        else:
            # Si se agotan los reintentos
            raise Exception(f"Error persistente: no se pudo descargar {symbol} {interval}")

def save_raw_data(
    klines: list,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: Union[str, None],
    base_output_dir: str
) -> str:
    """
    Convierte la lista de klines a DataFrame, lo guarda en formato Parquet
    en la carpeta raw_data/ dentro de base_output_dir/SYMBOL.

    Retorna la ruta (path) del archivo generado.
    """
    logger = logging.getLogger("download_data.binance")

    columns = [
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)

    # Convertir timestamps a datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Convertir columnas numéricas
    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Eliminar columna 'ignore' si está presente
    if "ignore" in df.columns:
        df.drop(columns=["ignore"], inplace=True)

    # Construir path de salida: base_output_dir/SYMBOL/raw_data/
    symbol_dir = os.path.join(base_output_dir, symbol)
    raw_data_dir = os.path.join(symbol_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Generar nombre de archivo
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "") if end_date else "latest"
    filename = f"{symbol}_{interval}_{start_str}_to_{end_str}.parquet"
    output_path = os.path.join(raw_data_dir, filename)

    df.to_parquet(output_path, index=False)
    logger.info(f"Datos RAW guardados en: {output_path} - Filas: {len(df)}")

    return output_path
