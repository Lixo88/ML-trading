import logging
import pandas as pd
import numpy as np
from datetime import datetime
from prepare_data.logs import log_event

def identify_supply_demand_zones(df):
    """
    Identifica zonas de oferta y demanda basadas en momentum, volumen relativo y ATR.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos.
    
    Returns:
        pd.DataFrame: DataFrame con zonas de oferta y demanda marcadas.
    """
    log_event("INFO", "Iniciando identificación de zonas de oferta y demanda", module="zones")
    
    try:
        df['body'] = df['close'] - df['open']  # Cuerpo de la vela (positivo si es alcista, negativo si es bajista)
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()  # Volumen relativo
        df['atr'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()

        # Identificar zonas de demanda (soporte) → Velas alcistas con volumen alto
        demand_zones = (df['body'] > 0) & (df['body'] > df['body'].rolling(window=20).mean()) & (df['relative_volume'] > 1.5)
        df['demand_zone'] = np.where(demand_zones, df['low'] - df['atr'] * 0.5, np.nan)
        
        # Identificar zonas de oferta (resistencia) → Velas bajistas con volumen alto
        supply_zones = (df['body'] < 0) & (abs(df['body']) > abs(df['body'].rolling(window=20).mean())) & (df['relative_volume'] > 1.5)
        df['supply_zone'] = np.where(supply_zones, df['high'] + df['atr'] * 0.5, np.nan)

        log_event("INFO", f"Zonas de demanda detectadas: {df['demand_zone'].notna().sum()}", module="zones")
        log_event("INFO", f"Zonas de oferta detectadas: {df['supply_zone'].notna().sum()}", module="zones")
    
    except Exception as e:
        log_event("ERROR", f"Error en la identificación de zonas: {str(e)}", module="zones")
    
    return df

def log_zone_identification(df, timeframe):
    """
    Registra el proceso de identificación de zonas de oferta y demanda.
    """
    total_demand_zones = df['demand_zone'].notna().sum()
    total_supply_zones = df['supply_zone'].notna().sum()
    total_zones = total_demand_zones + total_supply_zones
    log_event(
        'info',
        f"Identificación de zonas completada para {timeframe}. "
        f"Zonas de demanda: {total_demand_zones}, "
        f"Zonas de oferta: {total_supply_zones}, Total: {total_zones}",
        module="zones"
    )

def log_zone_adjustments(df):
    """
    Registra el proceso de ajuste dinámico de zonas.
    """
    merged_demand_zones = df['demand_merged_zone'].sum()
    merged_supply_zones = df['supply_merged_zone'].sum()
    log_event(
        'info',
        f"Ajuste dinámico de zonas completado. "
        f"Zonas de demanda fusionadas: {merged_demand_zones}, "
        f"Zonas de oferta fusionadas: {merged_supply_zones}",
        module="zones"
    )

def log_zone_strength(df):  #debe fusionarse con zone_relevance en context.py
    """
    Registra las métricas de fuerza de las zonas.
    """
    for zone_type in ['demand', 'supply']:
        repetitions = df[f'{zone_type}_repetitions'].sum()
        total_volume = df[f'{zone_type}_volume'].sum()
        avg_impact = df[f'{zone_type}_impact'].mean()
        log_event(
            'info',
            (f"Fuerza de zonas ({zone_type}): "
             f"Repeticiones: {repetitions}, "
             f"Volumen total: {total_volume}, "
             f"Impacto promedio: {avg_impact:.4f}"),
            module="zones"
        )

def log_cross_temporal_validation(df_4h, df_1d):
    """
    Registra los resultados de la validación cruzada entre temporalidades.
    """
    validated_demand_zones = df_4h['demand_validated'].sum()
    validated_supply_zones = df_4h['supply_validated'].sum()
    log_event(
        'info',
        f"Validación cruzada completada. "
        f"Zonas de demanda validadas: {validated_demand_zones}, "
        f"Zonas de oferta validadas: {validated_supply_zones}",
        module="zones"
    )
