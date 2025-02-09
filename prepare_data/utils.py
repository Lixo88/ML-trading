import pandas as pd
import numpy as np
import time
import os
from prepare_data.logs import log_event

def resample_dataframe(df, timeframe):
    """
    Convierte un DataFrame a una nueva temporalidad.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos.
        timeframe (str): Temporalidad destino ('1H', '4H', '1D').
    
    Returns:
        pd.DataFrame: DataFrame resampleado.
    """
    log_event("INFO", f"Iniciando resampleo de datos a {timeframe}", module="utils")
    try:
        df_resampled = df.resample(timeframe, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        log_event("INFO", f"Resampleo a {timeframe} completado", module="utils")
        return df_resampled
    except Exception as e:
        log_event("ERROR", f"Error en resampleo a {timeframe}: {str(e)}", module="utils")
        return df

def convert_timestamp_to_datetime(timestamp):
    """
    Convierte un timestamp en milisegundos a un objeto datetime.
    """
    return pd.to_datetime(timestamp, unit='ms')

def normalize_data(df, column, method="minmax"):
    """
    Normaliza una columna específica del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con datos.
        column (str): Nombre de la columna a normalizar.
        method (str): Método de normalización ('minmax' o 'zscore').
    
    Returns:
        pd.DataFrame: DataFrame con la columna normalizada.
    """
    log_event("INFO", f"Normalizando columna {column} usando {method}", module="utils")
    try:
        if method == "minmax":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        elif method == "zscore":
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        log_event("INFO", f"Normalización de {column} completada", module="utils")
    except Exception as e:
        log_event("ERROR", f"Error al normalizar {column}: {str(e)}", module="utils")
    return df

def calculate_percentage_change(df, column="close"):
    """
    Calcula la variación porcentual de una columna.
    """
    df[f'{column}_pct_change'] = df[column].pct_change()
    return df

def calculate_price_delta(df, column="close"):
    """
    Calcula la diferencia absoluta de precio entre velas consecutivas.
    """
    df[f'{column}_delta'] = df[column].diff()
    return df

def time_execution(func):
    """
    Decorador para medir el tiempo de ejecución de una función.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_event("INFO", f"{func.__name__} ejecutada en {end_time - start_time:.4f} segundos", module="utils")
        return result
    return wrapper


def save_processed_data(df, output_filepath):
    """
    Guarda un DataFrame procesado en un archivo Parquet.

    Args:
        df (pd.DataFrame): DataFrame a guardar.
        output_filepath (str): Ruta del archivo de salida.
    """
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_parquet(output_filepath, index=False)
        log_event("INFO", f"Datos guardados en {output_filepath}", module="utils")
    except Exception as e:
        log_event("ERROR", f"Error al guardar datos en {output_filepath}: {str(e)}", module="utils")
