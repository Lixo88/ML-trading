# validation.py

import os
import pandas as pd
import logging
from datetime import timedelta

def validate_and_correct_data(filepath: str, base_output_dir: str) -> str:
    """
    Valida el archivo Parquet (OHLCV), corrige NaNs y gaps, y guarda
    un archivo 'validado' en la carpeta principal de cada símbolo.
    
    Retorna la ruta del archivo parquet validado.
    """
    logging.info(f"Validando archivo: {filepath}")
    df = pd.read_parquet(filepath)

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            msg = f"Falta la columna {col} en {filepath}"
            logging.error(msg)
            raise ValueError(msg)

    # Ordenar por timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Interpolar posibles NaNs en precios y volumen
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logging.warning(f"NaNs detectados en {filepath}: {nan_counts.to_dict()}")
        df[["open", "high", "low", "close", "volume"]] = (
            df[["open", "high", "low", "close", "volume"]].interpolate(method='linear')
        )
        # Si tras interpolar quedan NaNs, dropearlos
        df.dropna(inplace=True)

    # Detectar timeframe por nombre de archivo (heurística)
    timeframe = detect_timeframe(filepath)
    expected_hours = get_expected_interval_hours(timeframe)
    if expected_hours is None:
        logging.warning(f"No se pudo detectar timeframe en {filepath}, no se corrigen gaps.")
    else:
        df = correct_gaps(df, expected_hours)

    # Guardar en la carpeta principal del símbolo, con sufijo _validated
    symbol = detect_symbol_from_path(filepath)
    symbol_dir = os.path.join(base_output_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    output_filename = os.path.basename(filepath).replace(".parquet", "_validated.parquet")
    output_path = os.path.join(symbol_dir, output_filename)
    df.to_parquet(output_path, index=False)

    logging.info(f"Archivo validado guardado en: {output_path} - Filas: {len(df)}")

    return output_path

def detect_timeframe(filepath: str) -> str:
    """
    Dado el nombre del archivo, retorna la temporalidad ('1d','4h','1h','15m') o None.
    """
    filename = os.path.basename(filepath)
    if "1d" in filename:
        return "1d"
    elif "4h" in filename:
        return "4h"
    elif "1h" in filename:
        return "1h"
    elif "15m" in filename:
        return "15m"
    else:
        return None

def get_expected_interval_hours(timeframe: str) -> float:
    """
    Retorna el intervalo esperado en horas según la temporalidad.
    """
    mapping = {
        "1d": 24.0,
        "4h": 4.0,
        "1h": 1.0,
        "15m": 0.25
    }
    return mapping.get(timeframe, None)

def correct_gaps(df: pd.DataFrame, expected_hours: float) -> pd.DataFrame:
    """
    Detecta gaps donde la diferencia entre timestamps es mayor al esperado.
    Inserta filas vacías y las interpola para no romper la continuidad.
    """
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600.0
    gap_indices = df[df["time_diff"] > expected_hours].index

    if len(gap_indices) == 0:
        logging.info("No se detectaron gaps significativos.")
        df.drop(columns=["time_diff"], inplace=True)
        return df

    logging.warning(f"Se detectaron {len(gap_indices)} gaps con time_diff > {expected_hours} horas.")

    # Generar timestamps faltantes e interpolar
    missing_rows = []
    for i in gap_indices:
        # Timestamps de la fila previa y la actual
        prev_ts = df.loc[i-1, "timestamp"]
        curr_ts = df.loc[i, "timestamp"]
        # Generar timestamps intermedios a la frecuencia esperada
        freq_minutes = int(expected_hours * 60)
        # Arrancar un paso después de prev_ts
        new_range = pd.date_range(prev_ts + timedelta(minutes=freq_minutes),
                                  curr_ts - timedelta(minutes=freq_minutes),
                                  freq=f"{freq_minutes}T")
        if len(new_range) > 0:
            for ts in new_range:
                missing_rows.append({
                    "timestamp": ts,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                    "time_diff": None
                })

    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        # Interpolar
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].interpolate(method="linear")

    df.drop(columns=["time_diff"], inplace=True)
    return df

def detect_symbol_from_path(filepath: str) -> str:
    """
    Extrae el símbolo del nombre del archivo (simple heuristic).
    Asume que el nombre del archivo inicia con 'SYMBOL_interval_...'.
    """
    filename = os.path.basename(filepath)
    # ejemplo: "BTCUSDT_4h_20220101_to_20220701.parquet"
    # Tomar la primera parte antes del primer '_'
    symbol = filename.split("_")[0]
    return symbol
