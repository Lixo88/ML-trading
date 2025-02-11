import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from prepare_data.logs import log_event

# ----------------------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------------------

def detect_swing_lows(df: pd.DataFrame, window=3) -> pd.Series:
    rolling_min = df['low'].rolling(window=2*window+1, center=True).min()
    return (df['low'] == rolling_min)

def detect_swing_highs(df: pd.DataFrame, window=3) -> pd.Series:
    rolling_max = df['high'].rolling(window=2*window+1, center=True).max()
    return (df['high'] == rolling_max)

def placeholder_zone_strength_model(features: pd.DataFrame, penalty=0.0) -> np.ndarray:
    """
    Ejemplo lineal con 4 factores principales:
      zone_strength = 0.4*repetitions + 0.3*volume_score + 0.2*fib_factor + 0.1*delta_vol
    Luego penalizamos la frescura: zone_strength -= penalty * zone_freshness
    """
    w_rep, w_vol, w_fib, w_delta = 0.4, 0.3, 0.2, 0.1
    reps = features['repetitions']
    vol_score = features['volume_score']
    fib_factor = features.get('fib_0.382_overlap', 0.0)
    delta_vol  = features.get('volume_delta_score', 0.0)
    base_strength = (w_rep*reps + w_vol*vol_score + w_fib*fib_factor + w_delta*delta_vol)

    if 'zone_freshness' in features.columns:
        freshness = features['zone_freshness']
        # penalizamos
        final_strength = base_strength - penalty*freshness
    else:
        final_strength = base_strength

    return final_strength

# ----------------------------------------------------------------
# 1) IDENTIFICAR ZONAS
# ----------------------------------------------------------------
def identify_zones(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    1) Detectar pivots (swing_low/high)
    2) Filtro de volumen (relative_volume >= volume_threshold)
    3) Filtro de mecha (mecha_thr_demand / mecha_thr_supply)
    4) Filtro de body ratio (body_thr_demand / body_thr_supply)
    5) Asignar demand_zone_lower/upper o supply_zone_lower/upper
    """
    log_event("INFO", "Iniciando identify_zones()...", module="zones")

    # --- Leer config ---
    pivot_window      = config.get('pivot_window', 3)
    vol_thr           = config.get('volume_threshold', 1.5)
    atr_mult          = config.get('atr_multiplier', 0.5)
    mecha_thr_dem     = config.get('mecha_thr_demand', 0.2)
    mecha_thr_sup     = config.get('mecha_thr_supply', 0.2)
    body_thr_dem      = config.get('body_thr_demand', 0.0)
    body_thr_sup      = config.get('body_thr_supply', 0.0)

    log_event("INFO",
        (f"Params => pivot_window={pivot_window}, volume_threshold={vol_thr}, "
         f"atr_multiplier={atr_mult}, mechaD={mecha_thr_dem}, mechaS={mecha_thr_sup}, "
         f"bodyD={body_thr_dem}, bodyS={body_thr_sup}"),
        module="zones")

    # --- Calcular mechas si no existen ---
    if 'lower_wick_ratio' not in df.columns or 'upper_wick_ratio' not in df.columns:
        df['total_range'] = (df['high'] - df['low']).replace(0, np.nan)
        df['lower_wick'] = (df['open'] - df['low']).clip(lower=0)
        df['upper_wick'] = (df['high'] - df['close']).clip(lower=0)
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']

    # --- Calcular body ratio ---
    # Ej: body_ratio = abs(close-open)/ (high-low)
    if 'body_ratio' not in df.columns:
        body_abs = (df['close'] - df['open']).abs()
        body_ratio = body_abs / df['total_range'].replace(0, np.nan)
        df['body_ratio'] = body_ratio.fillna(0)

    # --- Detectar swing lows/highs ---
    df['is_swing_low']  = detect_swing_lows(df, window=pivot_window)
    df['is_swing_high'] = detect_swing_highs(df, window=pivot_window)

    # --- Filtrar volumen (relative_volume) ---
    if 'relative_volume' not in df.columns:
        rv = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
        df['relative_volume'] = rv.fillna(0)

    # Demand => is_swing_low + vol >= thr + mecha + body
    demand_mask = (
        df['is_swing_low'] &
        (df['relative_volume'] >= vol_thr) &
        (df['lower_wick_ratio'] <= mecha_thr_dem) &
        (df['body_ratio'] >= body_thr_dem)
    )

    # Supply => is_swing_high + vol >= thr + mecha + body
    supply_mask = (
        df['is_swing_high'] &
        (df['relative_volume'] >= vol_thr) &
        (df['upper_wick_ratio'] <= mecha_thr_sup) &
        (df['body_ratio'] >= body_thr_sup)
    )

    # ATR
    if 'ATR' not in df.columns:
        log_event("WARNING", "No ATR column found, fallback=0", module="zones")
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

    num_demand = demand_mask.sum()
    num_supply = supply_mask.sum()
    log_event("INFO", f"identify_zones() => demand pivots={num_demand}, supply pivots={num_supply}, total={num_demand+num_supply}", module="zones")

    return df

def cluster_zones(df: pd.DataFrame, zone_type: str, config: dict) -> pd.DataFrame:
    """
    Aplica DBSCAN en log-precio para unificar pivots. Retorna un DF con zone_lower, zone_upper, etc.
    """
    eps = config.get('dbscan_eps', 0.02)
    min_samp = config.get('dbscan_min_samples', 1)

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

    prices_log = np.log(zone_df[pivot_col].values.reshape(-1,1))
    clustering = DBSCAN(eps=eps, min_samples=min_samp).fit(prices_log)
    zone_df['cluster_id'] = clustering.labels_

    outliers_count = (zone_df['cluster_id'] == -1).sum()
    log_event("INFO", f"cluster_zones({zone_type}): outliers={outliers_count}", module="zones")

    merged_rows = []
    for cid, grp in zone_df.groupby('cluster_id'):
        if cid == -1:
            continue
        pivot_mean = grp[pivot_col].mean()
        z_lower = grp[zone_lower_col].min()
        z_upper = grp[zone_upper_col].max()
        rep_sum = grp[reps_col].sum()
        vol_sum = grp[vol_col].sum()

        last_ts = grp['timestamp'].max()
        main_max_ts = df['timestamp'].max() if 'timestamp' in df.columns else last_ts
        zone_age = 0.0
        if pd.notnull(last_ts) and pd.notnull(main_max_ts):
            zone_age = (main_max_ts - last_ts).total_seconds()/3600.0

        merged_rows.append({
            'cluster_id': cid,
            'pivot_mean_price': pivot_mean,
            'zone_lower': z_lower,
            'zone_upper': z_upper,
            'repetitions': rep_sum,
            'volume_score': vol_sum,
            'last_pivot_ts': last_ts,
            'zone_freshness': zone_age
        })

    merged_df = pd.DataFrame(merged_rows)
    log_event("INFO", f"cluster_zones({zone_type}): total clusters => {len(merged_df)}", module="zones")
    return merged_df

def calculate_zone_strength(merged_df: pd.DataFrame, df_main: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Genera 'zone_raw_strength' usando un placeholder sub-model.
    Aplica penalización de frescura si freshness_penalty>0.
    """
    if merged_df.empty:
        merged_df['zone_raw_strength'] = []
        return merged_df

    # Usar fib y volume_delta si config lo habilita
    use_fib = config.get('use_fib', True)
    fib_factor = 0.0
    if use_fib and 'fib_0.382' in df_main.columns:
        fib_factor = df_main['fib_0.382'].mean()

    use_vol_delta = config.get('use_volume_delta', True)
    vol_delta_score = 0.0
    if use_vol_delta and 'volume_delta' in df_main.columns:
        vol_delta_score = df_main['volume_delta'].mean()

    features = merged_df[['repetitions','volume_score','zone_freshness']].copy()
    features['fib_0.382_overlap'] = fib_factor
    features['volume_delta_score'] = vol_delta_score

    # penalización
    freshness_penalty = config.get('freshness_penalty', 0.0)
    zone_strength = placeholder_zone_strength_model(features, penalty=freshness_penalty)

    merged_df['zone_raw_strength'] = zone_strength
    return merged_df

def run_zones_pipeline(df: pd.DataFrame, config: dict):
    """
    Orquesta la detección de zonas:
      1) identify_zones -> pivots en df_main (vela-based)
      2) cluster_zones(demand) + calculate_zone_strength
      3) cluster_zones(supply) + calculate_zone_strength
      4) Concat -> df_zones
    Retorna (df_main, df_zones).
    """
    log_event("INFO", "Iniciando run_zones_pipeline()...", module="zones")

    df_main = identify_zones(df, config)

    demand_merged = cluster_zones(df_main, 'demand', config)
    demand_merged = calculate_zone_strength(demand_merged, df_main, config)
    demand_merged['zone_type'] = 'demand'

    supply_merged = cluster_zones(df_main, 'supply', config)
    supply_merged = calculate_zone_strength(supply_merged, df_main, config)
    supply_merged['zone_type'] = 'supply'

    df_zones = pd.concat([demand_merged, supply_merged], ignore_index=True)

    log_event("INFO", f"run_zones_pipeline => demand_clusters={len(demand_merged)}, supply_clusters={len(supply_merged)}", module="zones")

    # Podemos loguear un summary de zone_raw_strength
    if not df_zones.empty:
        mean_strength = df_zones['zone_raw_strength'].mean()
        log_event("INFO", f"Avg zone_raw_strength={mean_strength:.2f}", module="zones")

    log_event("INFO", "run_zones_pipeline completado.", module="zones")
    return df_main, df_zones
