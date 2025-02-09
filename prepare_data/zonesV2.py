"""
zones.py

Versión extendida con:
 - Filtro de mecha en pivots (lower_wick_ratio / upper_wick_ratio).
 - Cálculo de freshness (edad de la zona).
 - Logs ampliados sobre pivots, outliers, etc.

Retorna (df_main, df_zones):
   df_main -> candle-based (1 fila por vela) con cols de pivots.
   df_zones -> 1 fila por zona unificada (demand/supply).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from prepare_data.logs import log_event

DEFAULT_CONFIG = {
    'pivot_window': 3,
    'volume_threshold': 1.5,
    'atr_multiplier': 0.5,
    'dbscan_eps': 0.02,       # 2% en log scale
    'dbscan_min_samples': 1,
    'use_fib': True,
    'use_volume_delta': True,
    'zone_strength_model': None,
    # Filtro de mecha
    'mecha_thr_demand': 0.2,  # ratio <= 20% => candle con mecha inferior "corta"
    'mecha_thr_supply': 0.2,  # ratio <= 20% => candle con mecha superior "corta"
}

# ----------------------------------------------------------------
# FUNCIONES AUX
# ----------------------------------------------------------------

def detect_swing_lows(df: pd.DataFrame, window=3) -> pd.Series:
    rolling_min = df['low'].rolling(window=2*window+1, center=True).min()
    return (df['low'] == rolling_min)

def detect_swing_highs(df: pd.DataFrame, window=3) -> pd.Series:
    rolling_max = df['high'].rolling(window=2*window+1, center=True).max()
    return (df['high'] == rolling_max)

def placeholder_zone_strength_model(features: pd.DataFrame) -> np.ndarray:
    # Ejemplo lineal:
    # zone_strength = 0.4*repetitions + 0.3*volume_score + 0.2*fib_factor + 0.1*delta_vol
    w_rep, w_vol, w_fib, w_delta = 0.4, 0.3, 0.2, 0.1
    reps = features['repetitions']
    vol_score = features['volume_score']
    fib_factor = features.get('fib_0.382_overlap', 0.0)
    delta_vol  = features.get('volume_delta_score', 0.0)
    return (w_rep*reps + w_vol*vol_score + w_fib*fib_factor + w_delta*delta_vol)

# ----------------------------------------------------------------
# 1) IDENTIFICAR ZONAS
# ----------------------------------------------------------------
def identify_zones(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    - Detecta swing lows/highs con window.
    - Filtro de volumen relative >= volume_threshold.
    - Filtro de mecha (opcional).
    - Define col demand_zone_lower/upper, supply_zone_lower/upper.
    - Retorna df con dichas cols.
    """
    if config is None:
        config = DEFAULT_CONFIG

    log_event("INFO", "Iniciando identify_zones()...", module="zones")

    window         = config['pivot_window']
    vol_thr        = config['volume_threshold']
    atr_mult       = config['atr_multiplier']
    mecha_thr_dem  = config['mecha_thr_demand']
    mecha_thr_sup  = config['mecha_thr_supply']

    # Calcular mechas si no existen
    if 'lower_wick_ratio' not in df.columns or 'upper_wick_ratio' not in df.columns:
        # lower_wick = open - low, upper_wick = high - close, etc. O define tu ratio preferido
        # A modo de ejemplo:
        df['total_range'] = (df['high'] - df['low']).replace(0, np.nan)  # para evitar div/0
        df['lower_wick'] = (df['open'] - df['low']).clip(lower=0)
        df['upper_wick'] = (df['high'] - df['close']).clip(lower=0)
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']

    # Detectar pivot lows/highs
    df['is_swing_low']  = detect_swing_lows(df, window=window)
    df['is_swing_high'] = detect_swing_highs(df, window=window)

    # Filtrar volumen
    if 'relative_volume' not in df.columns:
        rv = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
        df['relative_volume'] = rv.fillna(0)

    # Demand: swing_low + vol >= vol_thr + mecha inferior <= mecha_thr_dem
    demand_mask = (
        df['is_swing_low'] &
        (df['relative_volume'] >= vol_thr) &
        (df['lower_wick_ratio'] <= mecha_thr_dem)
    )
    # Supply: swing_high + vol >= vol_thr + mecha superior <= mecha_thr_sup
    supply_mask = (
        df['is_swing_high'] &
        (df['relative_volume'] >= vol_thr) &
        (df['upper_wick_ratio'] <= mecha_thr_sup)
    )

    # ATR
    if 'ATR' not in df.columns:
        log_event("WARNING", "No ATR column found, fallback to 0", module="zones")
        df['ATR'] = 0

    # Demand zone
    df['demand_zone_lower'] = np.where(demand_mask, df['low'] - atr_mult*df['ATR'], np.nan)
    df['demand_zone_upper'] = np.where(demand_mask, df['low'], np.nan)
    df['demand_repetitions'] = np.where(demand_mask, 1, 0)
    df['demand_volume']      = np.where(demand_mask, df['volume'], 0)

    # Supply zone
    df['supply_zone_lower'] = np.where(supply_mask, df['high'], np.nan)
    df['supply_zone_upper'] = np.where(supply_mask, df['high'] + atr_mult*df['ATR'], np.nan)
    df['supply_repetitions'] = np.where(supply_mask, 1, 0)
    df['supply_volume']      = np.where(supply_mask, df['volume'], 0)

    # Logging
    num_demand_pivots = demand_mask.sum()
    num_supply_pivots = supply_mask.sum()
    total_pivots = num_demand_pivots + num_supply_pivots
    log_event("INFO", f"identify_zones(): detectó {total_pivots} pivots totales => {num_demand_pivots} demand, {num_supply_pivots} supply", module="zones")

    return df

# ----------------------------------------------------------------
# 2) CLUSTER
# ----------------------------------------------------------------
def cluster_zones(df: pd.DataFrame, zone_type: str, config=None) -> pd.DataFrame:
    """
    - Toma pivots de 'zone_type' (demand/supply).
    - DBSCAN en log-precio => unifica cluster.
    - Retorna DF con [zone_lower, zone_upper, repetitions, volume_score, cluster_id, ...].
    """
    if config is None:
        config = DEFAULT_CONFIG

    eps = config['dbscan_eps']
    min_samp = config['dbscan_min_samples']

    if zone_type == 'demand':
        pivot_col     = 'demand_zone_upper'
        zone_lower_col= 'demand_zone_lower'
        zone_upper_col= 'demand_zone_upper'
        reps_col      = 'demand_repetitions'
        vol_col       = 'demand_volume'
    else:
        pivot_col     = 'supply_zone_lower'
        zone_lower_col= 'supply_zone_lower'
        zone_upper_col= 'supply_zone_upper'
        reps_col      = f'{zone_type}_repetitions'
        vol_col       = f'{zone_type}_volume'

    zone_df = df[[pivot_col, zone_lower_col, zone_upper_col, reps_col, vol_col, 'timestamp']].copy()
    zone_df.dropna(subset=[pivot_col], inplace=True)
    if zone_df.empty:
        log_event("INFO", f"cluster_zones({zone_type}): no pivots => DF vacío", module="zones")
        return pd.DataFrame()

    # DBSCAN
    prices_log = np.log(zone_df[pivot_col].values.reshape(-1,1))
    clustering = DBSCAN(eps=eps, min_samples=min_samp).fit(prices_log)
    zone_df['cluster_id'] = clustering.labels_

    # stats de outliers
    outliers_count = (zone_df['cluster_id'] == -1).sum()
    log_event("INFO", f"cluster_zones({zone_type}): outliers = {outliers_count}", module="zones")

    merged_rows = []
    for cid, grp in zone_df.groupby('cluster_id'):
        if cid == -1:
            continue
        pivot_mean = grp[pivot_col].mean()
        z_lower = grp[zone_lower_col].min()
        z_upper = grp[zone_upper_col].max()
        rep_sum = grp[reps_col].sum()
        vol_sum = grp[vol_col].sum()

        # Freshness => consideramos timestamp de pivot + zone age
        # Podríamos usar la última (max) o la primera (min) pivot de ese cluster
        last_ts = grp['timestamp'].max()
        # Ej: definimos freshness = [ df['timestamp'].max() - last_ts ] en días
        main_max_ts = df['timestamp'].max() if 'timestamp' in df.columns else last_ts
        zone_age = (main_max_ts - last_ts).total_seconds() / 3600.0  if not pd.isnull(last_ts) else 0.0
        # lo dejamos en horas, se puede convertir a días

        merged_rows.append({
            'cluster_id': cid,
            'pivot_mean_price': pivot_mean,
            'zone_lower': z_lower,
            'zone_upper': z_upper,
            'repetitions': rep_sum,
            'volume_score': vol_sum,
            'last_pivot_ts': last_ts,
            'zone_freshness': zone_age  # mayor => zona más vieja
        })

    merged_df = pd.DataFrame(merged_rows)
    log_event("INFO", f"cluster_zones({zone_type}): total clusters => {len(merged_df)}", module="zones")

    return merged_df

# ----------------------------------------------------------------
# 3) CALCULAR zone_raw_strength
# ----------------------------------------------------------------
def calculate_zone_strength(merged_df: pd.DataFrame, df_main: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    - Se apoya en fib, volume_delta si config lo habilita, añade 'zone_raw_strength'.
    - Usa un placeholder sub-modelo o uno real.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if merged_df.empty:
        merged_df['zone_raw_strength'] = []
        return merged_df

    # factor fib
    fib_factor = 0.0
    if config['use_fib'] and 'fib_0.382' in df_main.columns:
        fib_factor = df_main['fib_0.382'].mean()  # placeholder

    vol_delta_score = 0.0
    if config['use_volume_delta'] and 'volume_delta' in df_main.columns:
        vol_delta_score = df_main['volume_delta'].mean()

    features = merged_df[['repetitions','volume_score','zone_freshness']].copy()
    # "freshness" se podría usar en el sub-modelo => p.e. features['freshness'] = merged_df['zone_freshness']
    # En este placeholder, no la usamos directamente. Podrías sumarla con un weight.

    features['fib_0.382_overlap'] = fib_factor
    features['volume_delta_score'] = vol_delta_score

    # Llamar sub-modelo real vs. placeholder
    zone_strength = placeholder_zone_strength_model(features)

    # Como ejemplo, podríamos penalizar zonas muy viejas
    # zone_strength = zone_strength - 0.01*merged_df['zone_freshness'] (opcional)

    merged_df['zone_raw_strength'] = zone_strength
    return merged_df

# ----------------------------------------------------------------
# 4) RUN ZONES => retorna (df_main, df_zones)
# ----------------------------------------------------------------
def run_zones_pipeline(df: pd.DataFrame, config=None):
    """
    1) identify_zones => pivots en df_main
    2) cluster_zones(demand) + calculate_zone_strength => df_demand_zones
    3) cluster_zones(supply) + calculate_zone_strength => df_supply_zones
    4) Unir => df_zones
    5) Retorna (df_main, df_zones)
    """
    if config is None:
        config = DEFAULT_CONFIG

    log_event("INFO", "Iniciando run_zones_pipeline()...", module="zones")
    df_main = identify_zones(df, config)

    demand_merged = cluster_zones(df_main, 'demand', config)
    demand_merged = calculate_zone_strength(demand_merged, df_main, config)
    demand_merged['zone_type'] = 'demand'

    supply_merged = cluster_zones(df_main, 'supply', config)
    supply_merged = calculate_zone_strength(supply_merged, df_main, config)
    supply_merged['zone_type'] = 'supply'

    df_zones = pd.concat([demand_merged, supply_merged], ignore_index=True)

    num_demand_clusters = len(demand_merged)
    num_supply_clusters = len(supply_merged)
    log_event("INFO", f"run_zones_pipeline => demand_clusters={num_demand_clusters}, supply_clusters={num_supply_clusters}", module="zones")

    log_event("INFO", "run_zones_pipeline completado.", module="zones")
    return df_main, df_zones
