# prepare_data/labeling.py

import pandas as pd
from prepare_data.logs import log_event

def label_events(df):
    """
    Etiqueta eventos clave basados en interacciones con zonas de oferta y demanda.

    Args:
        df (pd.DataFrame): DataFrame con zonas identificadas y procesadas.

    Returns:
        pd.DataFrame: DataFrame con columnas adicionales para etiquetas de eventos.
    """
    df['event'] = 'none'
    df['event_type'] = 'none'

    for i in range(len(df)):
        # Eventos relacionados con zonas de demanda
        if pd.notna(df['demand_zone_lower'].iloc[i]) and df['low'].iloc[i] <= df['demand_zone_upper'].iloc[i]:
            df.loc[i, 'event'] = 'demand_interaction'
            if df['close'].iloc[i] > df['demand_zone_upper'].iloc[i]:
                df.loc[i, 'event_type'] = 'breakout_demand'
            elif df['low'].iloc[i] >= df['demand_zone_lower'].iloc[i]:
                df.loc[i, 'event_type'] = 'rebound_demand'

        # Eventos relacionados con zonas de oferta
        if pd.notna(df['supply_zone_upper'].iloc[i]) and df['high'].iloc[i] >= df['supply_zone_lower'].iloc[i]:
            df.loc[i, 'event'] = 'supply_interaction'
            if df['close'].iloc[i] < df['supply_zone_lower'].iloc[i]:
                df.loc[i, 'event_type'] = 'breakout_supply'
            elif df['high'].iloc[i] <= df['supply_zone_upper'].iloc[i]:
                df.loc[i, 'event_type'] = 'rebound_supply'

        # Ruptura fallida: Precio intenta romper pero regresa
        if pd.notna(df['demand_zone_upper'].iloc[i]) and df['close'].iloc[i] > df['demand_zone_upper'].iloc[i]:
            if i + 1 < len(df) and df['close'].iloc[i + 1] < df['demand_zone_upper'].iloc[i]:
                df.loc[i, 'event'] = 'failed_breakout'
                df.loc[i, 'event_type'] = 'failed_breakout_demand'

        if pd.notna(df['supply_zone_lower'].iloc[i]) and df['close'].iloc[i] < df['supply_zone_lower'].iloc[i]:
            if i + 1 < len(df) and df['close'].iloc[i + 1] > df['supply_zone_lower'].iloc[i]:
                df.loc[i, 'event'] = 'failed_breakout'
                df.loc[i, 'event_type'] = 'failed_breakout_supply'

        # Consolidación: Precio dentro de zonas pero sin rompimientos
        if (pd.notna(df['demand_zone_lower'].iloc[i]) and pd.notna(df['supply_zone_upper'].iloc[i])):
            if (df['low'].iloc[i] > df['demand_zone_lower'].iloc[i] and
                df['high'].iloc[i] < df['supply_zone_upper'].iloc[i]):
                df.loc[i, 'event'] = 'consolidation'
                df.loc[i, 'event_type'] = 'inside_range'

    log_event('info', f"Eventos etiquetados: {df['event'].value_counts().to_dict()}.")
    return df

def add_event_metadata(df):
    """
    Añade metadatos adicionales a los eventos clave para análisis posterior.

    Args:
        df (pd.DataFrame): DataFrame con eventos etiquetados.

    Returns:
        pd.DataFrame: DataFrame con columnas adicionales de metadatos.
    """
    df['event_momentum'] = df['close'] - df['open']
    df['event_duration'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['event_volatility'] = df['high'] - df['low']

    log_event('info', "Metadatos de eventos añadidos.")
    return df

def label_zone_strength(df):
    """
    Etiqueta la fuerza de las zonas basándose en métricas de repetición, volumen e impacto.

    Args:
        df (pd.DataFrame): DataFrame con zonas identificadas.

    Returns:
        pd.DataFrame: DataFrame con columna adicional para la fuerza de la zona.
    """
    def classify_strength(row):
        score = (0.4 * row['demand_repetitions'] + 
                 0.4 * row['demand_volume'] + 
                 0.2 * row['demand_impact'])
        if score > 15:
            return 'very_strong'
        elif score > 10:
            return 'strong'
        elif score > 5:
            return 'moderate'
        else:
            return 'weak'

    df['zone_strength'] = df.apply(classify_strength, axis=1)

    log_event('info', "Fuerza de las zonas etiquetada.")
    return df

def label_zone_type(df):
    """
    Etiqueta las zonas como soporte o resistencia según su ubicación y relevancia.

    Args:
        df (pd.DataFrame): DataFrame con zonas identificadas.

    Returns:
        pd.DataFrame: DataFrame con columna adicional para el tipo de zona.
    """
    df['zone_type'] = 'none'
    df['zone_status'] = 'inactive'

    # Etiquetar soporte y resistencia según la fuerza de la zona
    df.loc[df['zone_strength'].isin(['strong', 'very_strong']) & df['demand_zone_lower'].notna(), 'zone_type'] = 'support'
    df.loc[df['zone_strength'].isin(['strong', 'very_strong']) & df['demand_zone_lower'].notna(), 'zone_status'] = 'recent'

    df.loc[df['zone_strength'].isin(['strong', 'very_strong']) & df['supply_zone_upper'].notna(), 'zone_type'] = 'resistance'
    df.loc[df['zone_strength'].isin(['strong', 'very_strong']) & df['supply_zone_upper'].notna(), 'zone_status'] = 'recent'

    # Etiquetas para zonas reactivadas
    df.loc[df['zone_strength'] == 'moderate', 'zone_status'] = 'reactivated'

    log_event('info', "Zonas etiquetadas como soporte o resistencia con estado dinámico.")
    return df

def label_accumulation_distribution(df):
    """
    Etiqueta eventos de acumulación y distribución dentro de las zonas.

    Args:
        df (pd.DataFrame): DataFrame con zonas y eventos procesados.

    Returns:
        pd.DataFrame: DataFrame con nuevas etiquetas para acumulación y distribución.
    """
    df['accumulation'] = False
    df['distribution'] = False

    for i in range(len(df)):
        # Acumulación: Velas pequeñas con bajo volumen en zona de demanda
        if pd.notna(df['demand_zone_lower'].iloc[i]):
            if (df['high'].iloc[i] - df['low'].iloc[i] < 0.5 * df['atr'].iloc[i] and
                df['volume'].iloc[i] < df['volume'].rolling(window=14).mean().iloc[i]):
                df.loc[i, 'accumulation'] = True

        # Distribución: Velas con alto volumen en zona de oferta
        if pd.notna(df['supply_zone_upper'].iloc[i]):
            if (df['volume'].iloc[i] > df['volume'].rolling(window=14).mean().iloc[i] and
                df['high'].iloc[i] <= df['supply_zone_upper'].iloc[i]):
                df.loc[i, 'distribution'] = True

    log_event('info', "Acumulación y distribución etiquetadas.")
    return df

def label_zone_impact(df):
    """
    Clasifica el impacto de las zonas basándose en su relevancia relativa.

    Args:
        df (pd.DataFrame): DataFrame con zonas procesadas.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna para clasificación de impacto.
    """
    def calculate_impact(row):
        impact_score = (0.4 * row['demand_repetitions'] + 
                        0.3 * row['volume'] / row['volume'].rolling(window=14).mean() + 
                        0.2 * (row['atr'] / row['close']) + 
                        0.1 * row['zone_strength_score'])
        if impact_score > 1.5:
            return 'high'
        elif impact_score > 0.8:
            return 'medium'
        else:
            return 'low'

    df['zone_impact'] = df.apply(calculate_impact, axis=1)

    log_event('info', "Impacto de las zonas clasificado.")
    return df

def finalize_labels(df):
    """
    Refina y valida las etiquetas para garantizar consistencia.

    Args:
        df (pd.DataFrame): DataFrame con eventos y metadatos etiquetados.

    Returns:
        pd.DataFrame: DataFrame finalizado con etiquetas refinadas.
    """
    # Eliminar filas sin eventos relevantes
    df = df[df['event'] != 'none']

    # Validación de etiquetas
    valid_events = ['demand_interaction', 'supply_interaction', 'consolidation', 'failed_breakout']
    df = df[df['event'].isin(valid_events)]

    log_event('info', f"Etiquetas refinadas y validadas. Total eventos relevantes: {len(df)}.")
    return df
