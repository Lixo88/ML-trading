"""
context.py

Objetivos:
 1) Validar zonas entre temporalidades (ej. 4H vs 1D) usando vectorized approach (cross join).
 2) (Opcional) Integrar key_events (placeholder).
 3) Asignar la zona más fuerte a cada vela en df_main (si config lo indica).
 4) Calcular zone_relevance (sub-model lineal placeholder o real).
Retorna (df_main_context, df_zones_context).
"""

import pandas as pd
import numpy as np
from prepare_data.logs import log_event

# ---------------------------------------------------------------------
# Funciones para Validar multi-timeframe via cross join
# ---------------------------------------------------------------------

def validate_cross_temporal_zones(df_zones_lower: pd.DataFrame,
                                  df_zones_higher: pd.DataFrame,
                                  config: dict) -> pd.DataFrame:
    """
    Versión vectorizada (cross join) para ver solapamiento:
     overlap_mask = (zone_lower_L <= zone_upper_H*(1+tol)) & (zone_upper_L >= zone_lower_H*(1-tol))
    Si hay solape => validated_by_higher = True.
    """
    tol = config.get('overlap_tolerance', 0.02)
    if df_zones_lower.empty or df_zones_higher.empty:
        log_event("INFO", "validate_cross_temporal_zones => uno de los DF está vacío. Sin validación.", module="context")
        df_zones_lower['validated_by_higher'] = False
        return df_zones_lower

    log_event("INFO", "Iniciando validación multi-TF (cross join vectorized).", module="context")

    # Filtrar supply vs demand por separate si deseas, o haremos un approach unificado
    # Approach: cross join por zone_type
    # -> primero separemos demand/supply en each DF

    # Example: 
    # Podríamos hacer un concat approach, pero lo haremos a lo simple.

    # We'll keep zone_type the same. Then merge where zone_type matches
    dfL = df_zones_lower.copy()
    dfL['key'] = 1
    dfH = df_zones_higher.copy()
    dfH['key'] = 1

    cross = dfL.merge(dfH, on='key', suffixes=('_L','_H'))
    # Filtrar por zone_type
    same_type_mask = (cross['zone_type_L'] == cross['zone_type_H'])
    cross = cross[same_type_mask]

    # Overlap cond:
    overlap_mask = (
        (cross['zone_lower_L'] <= cross['zone_upper_H']*(1+tol)) &
        (cross['zone_upper_L'] >= cross['zone_lower_H']*(1-tol))
    )
    cross_overlap = cross[overlap_mask]

    # cluster_id_L que se solapa => validated_by_higher = True
    validated_ids = cross_overlap['cluster_id_L'].unique()

    # Asignamos
    df_zones_lower['validated_by_higher'] = df_zones_lower['cluster_id'].isin(validated_ids)

    log_event("INFO",
              f"validate_cross_temporal_zones => Zonas validadas: {(df_zones_lower['validated_by_higher']).sum()} "
              f"de {len(df_zones_lower)}",
              module="context")

    return df_zones_lower

# ---------------------------------------------------------------------
# Placeholder para key_events
# ---------------------------------------------------------------------

def integrate_key_events(df_zones: pd.DataFrame, key_events, config: dict) -> pd.DataFrame:
    """
    Recibe df_zones y key_events (lista de dict, por ejemplo),
    asigna event_impact. Dejamos placeholder. 
    """
    use_key_events = config.get('use_key_events', False)
    if not use_key_events or not key_events:
        log_event("INFO", "integrate_key_events => Sin key events, event_impact=0.", module="context")
        df_zones['event_impact'] = 0.0
        return df_zones

    log_event("INFO", f"Integrando {len(key_events)} eventos en df_zones...", module="context")

    # Aquí haríamos un cross join o un approach con last_pivot_ts
    # De momento, placeholder => event_impact=0

    df_zones['event_impact'] = 0.0  # se podría decaer en ± config['decay_hours']
    return df_zones

# ---------------------------------------------------------------------
# Calcular zone_relevance
# ---------------------------------------------------------------------

def calculate_zone_relevance(df_zones: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Combina zone_raw_strength con event_impact + validación multiTF => zone_relevance.
    Ejemplo lineal:
      zone_relevance = zone_raw_strength + 0.3*event_impact + 1*(validated_by_higher)
    """
    if df_zones.empty:
        df_zones['zone_relevance'] = []
        return df_zones

    if 'zone_raw_strength' not in df_zones.columns:
        df_zones['zone_raw_strength'] = 0.0
    if 'event_impact' not in df_zones.columns:
        df_zones['event_impact'] = 0.0
    if 'validated_by_higher' not in df_zones.columns:
        df_zones['validated_by_higher'] = False

    # placeholder lineal
    validated_factor = df_zones['validated_by_higher'].astype(float)
    df_zones['zone_relevance'] = (
        df_zones['zone_raw_strength']
        + 0.3 * df_zones['event_impact']
        + 1.0 * validated_factor
    )

    return df_zones

# ---------------------------------------------------------------------
# Asignar la zona más fuerte a cada vela (opcional)
# ---------------------------------------------------------------------

def assign_zones_to_candles(df_main: pd.DataFrame,
                            df_zones: pd.DataFrame,
                            config: dict) -> pd.DataFrame:
    """
    - Asigna a cada vela (row en df_main) la zona (en df_zones) que sea "más cercana" 
      o "en la que esté dentro" (dependiendo de config).
    - Crea col ej. 'closest_demand_relevance', 'closest_supply_relevance'.
    - Vectorized cross join + argmax approach.
    """
    if df_main.empty or df_zones.empty:
        log_event("INFO", "assign_zones_to_candles => uno de los DF vacío, skipping", module="context")
        return df_main

    mode = config.get('zone_assign_dist_mode', 'closest_mean')  # 'inside_zone' or 'closest_mean'
    log_event("INFO", f"Asignando zona a velas => mode={mode}", module="context")

    # Ej. cross join
    dfM = df_main.copy()
    dfM['key'] = 1
    dfZ = df_zones.copy()
    dfZ['key'] = 1

    cross = dfM.merge(dfZ, on='key', suffixes=('_m','_z'))
    cross.drop(columns=['key'], inplace=True)

    # Filtrar demand vs supply si quieres 2 col. 
    # Ej: creamos 2 col: 'closest_demand_relevance','closest_supply_relevance'

    if mode == 'inside_zone':
        # inside => (low >= zone_lower & high <= zone_upper)
        cross['inside_demand'] = (
            (cross['zone_type'] == 'demand') &
            (cross['low_m'] >= cross['zone_lower']) &
            (cross['high_m'] <= cross['zone_upper'])
        )
        cross['inside_supply'] = (
            (cross['zone_type'] == 'supply') &
            (cross['low_m'] >= cross['zone_lower']) &
            (cross['high_m'] <= cross['zone_upper'])
        )
        # luego se agrupa por (timestamp_m) y sacas la mayor relevancia
    else:
        # closest_mean => dist = |close_m - pivot_mean_price|
        cross['dist_to_zone'] = (cross['close_m'] - cross['pivot_mean_price']).abs()

    # Ej: sacamos subset demand vs supply
    cross_demand = cross[cross['zone_type']=='demand'].copy()
    cross_supply = cross[cross['zone_type']=='supply'].copy()

    # Para demand => groupby 'timestamp_m', quedarnos con la zona con menor dist
    if mode == 'closest_mean':
        # groupby min dist => la relev mas alta
        cross_demand.sort_values(by=['timestamp_m','dist_to_zone'], inplace=True)
        cross_demand = cross_demand.groupby('timestamp_m', as_index=False).first()  # coge la 1era (dist minima)
        cross_supply.sort_values(by=['timestamp_m','dist_to_zone'], inplace=True)
        cross_supply = cross_supply.groupby('timestamp_m', as_index=False).first()

        # ahora unimos
        df_main2 = df_main.merge(cross_demand[['timestamp_m','zone_relevance']],
                                 left_on='timestamp', right_on='timestamp_m', how='left')
        df_main2.rename(columns={'zone_relevance':'closest_demand_relevance'}, inplace=True)
        df_main2.drop(columns=['timestamp_m'], inplace=True)

        df_main2 = df_main2.merge(cross_supply[['timestamp_m','zone_relevance']],
                                  left_on='timestamp', right_on='timestamp_m', how='left')
        df_main2.rename(columns={'zone_relevance':'closest_supply_relevance'}, inplace=True)
        df_main2.drop(columns=['timestamp_m'], inplace=True)

        # logs
        log_event("INFO", "assign_zones_to_candles => modo closest_mean completado.", module="context")
        return df_main2

    elif mode == 'inside_zone':
        # Similar approach => filtrar inside_demand = True, groupby timestamp, 
        # quedarte con la mayor zone_relevance, etc.
        pass

    # si no hay modo...
    return df_main


# ---------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------

def run_context_pipeline(df_main: pd.DataFrame,
                         df_zones: pd.DataFrame,
                         # multiTF
                         df_zones_higher: pd.DataFrame=None,
                         # key events
                         key_events=None,
                         config: dict=None):
    """
    Orquesta:
     1) Validación multiTF => validated_by_higher
     2) integrate_key_events => event_impact
     3) calculate_zone_relevance => zone_relevance
     4) (opcional) assign_zones_to_candles => df_main con col 'closest_demand_relevance', etc.
    Retorna (df_main_context, df_zones_context)
    """
    if config is None:
        config = {}

    log_event("INFO", "Iniciando run_context_pipeline()...", module="context")

    df_main_ctx = df_main.copy()
    df_zones_ctx = df_zones.copy()

    # 1) multiTF
    multi_tf_val = config.get('multi_tf_validation', False)
    if multi_tf_val and df_zones_higher is not None and not df_zones_higher.empty:
        log_event("INFO", "validando cross-temporal: overlap 4H vs 1D...", module="context")
        df_zones_ctx = validate_cross_temporal_zones(df_zones_ctx, df_zones_higher, config)
    else:
        df_zones_ctx['validated_by_higher'] = False

    # 2) integrate key events (placeholder)
    df_zones_ctx = integrate_key_events(df_zones_ctx, key_events, config)

    # 3) zone_relevance
    df_zones_ctx = calculate_zone_relevance(df_zones_ctx, config)

    # 4) asignación de zona => if config says so
    if config.get('assign_zone_to_candles', False):
        df_main_ctx = assign_zones_to_candles(df_main_ctx, df_zones_ctx, config)

    log_event("INFO", "run_context_pipeline completado.", module="context")
    return df_main_ctx, df_zones_ctx
