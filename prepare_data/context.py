# prepare_data/context.py

import pandas as pd
import numpy as np
from prepare_data.logs import log_event

def validate_cross_temporal_zones(df_4h, df_1d):
    """
    Valida zonas de oferta y demanda entre temporalidades 4H y 1D.

    Args:
        df_4h (pd.DataFrame): DataFrame con zonas de 4H.
        df_1d (pd.DataFrame): DataFrame con zonas de 1D.

    Returns:
        pd.DataFrame, pd.DataFrame: DataFrames ajustados tras validación cruzada.
    """
    for zone_type in ['demand', 'supply']:
        lower_col_4h = f'{zone_type}_zone_lower'
        upper_col_4h = f'{zone_type}_zone_upper'

        lower_col_1d = f'{zone_type}_zone_lower'
        upper_col_1d = f'{zone_type}_zone_upper'

        df_4h[f'{zone_type}_validated'] = False
        for i in range(len(df_4h)):
            if pd.notna(df_4h[lower_col_4h].iloc[i]):
                overlap = (
                    (df_1d[lower_col_1d] <= df_4h[upper_col_4h].iloc[i]) &
                    (df_1d[upper_col_1d] >= df_4h[lower_col_4h].iloc[i])
                )
                if overlap.any():
                    df_4h.loc[i, f'{zone_type}_validated'] = True

    log_event('info', f"Validación cruzada completada entre 4H y 1D para {zone_type}.")
    return df_4h, df_1d

def integrate_key_events(df, key_events):
    """
    Incorpora eventos clave ("anclas") para ajustar dinámicamente la relevancia de las zonas.

    Args:
        df (pd.DataFrame): DataFrame con zonas identificadas.
        key_events (list of dict): Lista de eventos clave con formato:
            [{"timestamp": <datetime>, "event": <str>, "impact": <float>}]

    Returns:
        pd.DataFrame: DataFrame ajustado con eventos clave incorporados.
    """
    df['key_event'] = None
    df['event_impact'] = 0

    for event in key_events:
        event_time = event['timestamp']
        event_impact = event['impact']

        proximity_mask = (df['timestamp'] >= event_time - pd.Timedelta(hours=12)) & \
                         (df['timestamp'] <= event_time + pd.Timedelta(hours=12))

        df.loc[proximity_mask, 'key_event'] = event['event']
        df.loc[proximity_mask, 'event_impact'] += event_impact

    log_event('info', "Eventos clave integrados en las zonas.")
    return df

def calculate_zone_relevance(df):
    """
    Calcula la relevancia de las zonas combinando métricas de fuerza y eventos clave.

    Args:
        df (pd.DataFrame): DataFrame con zonas procesadas y eventos clave.

    Returns:
        pd.DataFrame: DataFrame con columna adicional de relevancia por zona.
    """
    for zone_type in ['demand', 'supply']:
        df[f'{zone_type}_relevance'] = (
            0.5 * df[f'{zone_type}_repetitions'] +
            0.3 * df[f'{zone_type}_volume'] +
            0.2 * df['event_impact']
        )

    log_event('info', "Relevancia de zonas calculada.")
    return df
