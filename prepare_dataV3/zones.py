"""
zones.py

Unificación de:
 - Pivot adaptativo con ATR
 - Mecha/body ratio (versión anterior)
 - Volumen configurable (zscore o relative)
 - DBSCAN multidimensional con transformaciones
 - zone_strength con ranking percentil y placeholders submodelo
 - Salida: df_main (vela-based con demand/supply columns) y df_zones (cluster-based)

Parámetros Principales en config['zones']:
 - pivot_window (min pivot window)
 - atr_multiplier
 - volume_approach: 'zscore' o 'relative'
 - volume_threshold / volume_zscore_threshold
 - mecha_thr_demand / mecha_thr_supply
 - body_thr_demand / body_thr_supply
 - dbscan_eps
 - min_samples_factor (e.g. 0.05 => 5% de pivots)
 - use_power_transform / use_quantile_transform (bool)
 - strength_weights (p.ej. [0.4, 0.3, 0.2]) => reps, vol, fresh
 - freshness_penalty => penalización a zone_age

Retorno:
 - df_main con columns 'demand_zone_lower/upper', 'supply_zone_lower/upper'
 - df_zones con cluster info, 'zone_strength', etc.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from prepare_data.logs import log_event, log_feature_stats

# ----------------------------------------------------------------
# 1) DETECT PIVOTS
# ----------------------------------------------------------------

def detect_pivots_adaptive(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Detecta pivots (demand/supply) usando ATR adaptativo y mecha/body ratio + volumen (zscore/relative).
    Crea en df las columnas:
     - demand_zone_lower, demand_zone_upper, supply_zone_lower, supply_zone_upper
    """
    module = "zones"
    log_event("INFO", "Iniciando detect_pivots_adaptive()...", module)

    # 1) Param config
    pivot_window  = config.get('pivot_window', 3)
    atr_mult      = config.get('atr_multiplier', 1.5)
    volume_approach = config.get('volume_approach', 'zscore')  # 'zscore' o 'relative'
    volume_thr   = config.get('volume_threshold', 1.5)         # si approach='relative'
    volume_z_thr = config.get('volume_zscore_threshold', 2.0)  # si approach='zscore'

    mecha_dem    = config.get('mecha_thr_demand', 0.2)
    mecha_sup    = config.get('mecha_thr_supply', 0.2)
    body_dem     = config.get('body_thr_demand', 0.0)
    body_sup     = config.get('body_thr_supply', 0.0)

    # Log de params
    log_event("INFO",
        (f"ATR-based pivot, pivot_window={pivot_window}, atr_mult={atr_mult}, "
         f"volume_approach={volume_approach}, volume_thr={volume_thr}, volume_z_thr={volume_z_thr}, "
         f"mechaD={mecha_dem}, mechaS={mecha_sup}, bodyD={body_dem}, bodyS={body_sup}"),
        module
    )

    df_out = df.copy()

    try:
        # 2) ATR adaptativo
        if 'ATR' not in df_out.columns:
            log_event("WARNING", "No ATR column found, fallback=1.0", module)
            df_out['ATR'] = 1.0

        rolling_atr_100 = df_out['ATR'].rolling(100).mean()
        avg_atr = rolling_atr_100.iloc[-1] if len(rolling_atr_100) >= 100 else 1.0
        current_atr = df_out['ATR'].iloc[-1] if len(df_out) > 0 else 1.0

        dynamic_window = max(
            pivot_window,
            min(10, int(np.ceil(5 * (current_atr / (avg_atr if avg_atr>1e-6 else 1e-6)))))
        )

        log_event("INFO", f"Dynamic pivot_window={dynamic_window}", module)

        # 3) Swing Lows/Highs con rolling
        #   rolling min centered => pivot
        df_out['roll_min'] = df_out['low'].rolling(window=2*dynamic_window+1, center=True).min()
        df_out['roll_max'] = df_out['high'].rolling(window=2*dynamic_window+1, center=True).max()

        df_out['is_swing_low']  = (df_out['low'] == df_out['roll_min'])
        df_out['is_swing_high'] = (df_out['high'] == df_out['roll_max'])

        # 4) Mecha/Body ratio si no existen
        if 'total_range' not in df_out.columns:
            df_out['total_range'] = (df_out['high'] - df_out['low']).replace(0, 1e-9)
        if 'lower_wick_ratio' not in df_out.columns or 'upper_wick_ratio' not in df_out.columns:
            df_out['lower_wick'] = (df_out['open'] - df_out['low']).clip(lower=0)
            df_out['upper_wick'] = (df_out['high'] - df_out['close']).clip(lower=0)
            df_out['lower_wick_ratio'] = df_out['lower_wick'] / df_out['total_range']
            df_out['upper_wick_ratio'] = df_out['upper_wick'] / df_out['total_range']
        if 'body_ratio' not in df_out.columns:
            body_abs = (df_out['close'] - df_out['open']).abs()
            df_out['body_ratio'] = (body_abs / df_out['total_range']).fillna(0)

        # 5) Volumen approach
        if volume_approach == 'zscore':
            if 'volume_zscore' not in df_out.columns:
                vol_mean = df_out['volume'].rolling(50).mean()
                vol_std  = df_out['volume'].rolling(50).std().replace(0,1e-6)
                df_out['volume_zscore'] = (df_out['volume'] - vol_mean) / vol_std
            vol_mask_dem = (df_out['volume_zscore'] >= volume_z_thr)
            vol_mask_sup = (df_out['volume_zscore'] >= volume_z_thr)
        else:
            # relative_volume
            if 'relative_volume' not in df_out.columns:
                rv = df_out['volume'] / (df_out['volume'].rolling(20).mean() + 1e-6)
                df_out['relative_volume'] = rv.fillna(0)
            vol_mask_dem = (df_out['relative_volume'] >= volume_thr)
            vol_mask_sup = (df_out['relative_volume'] >= volume_thr)

        # 6) Demand vs Supply Mask
        demand_mask = (
            df_out['is_swing_low'] &
            vol_mask_dem &
            (df_out['lower_wick_ratio'] <= mecha_dem) &
            (df_out['body_ratio'] >= body_dem)
        )
        supply_mask = (
            df_out['is_swing_high'] &
            vol_mask_sup &
            (df_out['upper_wick_ratio'] <= mecha_sup) &
            (df_out['body_ratio'] >= body_sup)
        )

        # 7) Asignar demand_zone_lower/upper / supply_zone_lower/upper
        df_out['demand_zone_lower'] = np.where(
            demand_mask, df_out['low'] - atr_mult*df_out['ATR'], np.nan
        )
        df_out['demand_zone_upper'] = np.where(
            demand_mask, df_out['low'], np.nan
        )

        df_out['supply_zone_lower'] = np.where(
            supply_mask, df_out['high'], np.nan
        )
        df_out['supply_zone_upper'] = np.where(
            supply_mask, df_out['high'] + atr_mult*df_out['ATR'], np.nan
        )

        num_dem = demand_mask.sum()
        num_sup = supply_mask.sum()
        log_event("INFO", f"Pivots adaptativos => Demand={num_dem}, Supply={num_sup}, total={num_dem+num_sup}", module)
        return df_out

    except Exception as e:
        log_event("ERROR", f"Error en detect_pivots_adaptive: {str(e)}", module)
        return df


def cluster_zones(df_main: pd.DataFrame, zone_type: str, config: dict) -> pd.DataFrame:
    """
    Aplica clustering en un espacio transformado (log price, vol, tiempo).
    - Toma pivots de 'demand_zone_upper' o 'supply_zone_lower', etc.
    - Usa Power/Quantile transform si config lo habilita.
    - DBSCAN adaptativo con min_samples = factor * n_pivots
    Retorna un DF clusterizado con zone_lower, zone_upper, last_pivot_ts, etc.
    """
    module = "zones"

    # Selecciona columnas segun zone_type
    if zone_type == 'demand':
        pivot_col  = 'demand_zone_upper'
        zone_low   = 'demand_zone_lower'
        zone_up    = 'demand_zone_upper'
    else:
        pivot_col  = 'supply_zone_lower'
        zone_low   = 'supply_zone_lower'
        zone_up    = 'supply_zone_upper'

    # Filtramos pivots
    zone_df = df_main[[pivot_col, zone_low, zone_up, 'timestamp']].copy()
    zone_df.dropna(subset=[pivot_col], inplace=True)

    if zone_df.empty:
        log_event("INFO", f"cluster_zones({zone_type}): no pivots => DF vacío", module)
        return pd.DataFrame()

    try:
        # 1) Armar feature matrix
        X = pd.DataFrame()
        # log price
        X['price_log'] = np.log(zone_df[pivot_col].values + 1e-9)
        # "tiempo" => index 
        # si 'timestamp' es datetime, convertimos a horas
        min_ts = zone_df['timestamp'].min()
        X['time_hrs'] = (zone_df['timestamp'] - min_ts).dt.total_seconds()/3600.0

        # Indicar un approach volumetrico? 
        # Si deseas, se puede usar 'relative_volume' mean en df_main. 
        # Placeholder: "vol_metric" = 1.0
        X['vol_metric'] = 1.0  # o zero if no logic

        use_power  = config.get('use_power_transform', True)
        use_quant  = config.get('use_quantile_transform', True)

        # 2) Transform
        X_trans = X.copy()
        if use_quant:
            qt = QuantileTransformer()
            X_trans = pd.DataFrame(qt.fit_transform(X_trans), columns=X.columns)
        if use_power:
            pt = PowerTransformer()
            X_trans = pd.DataFrame(pt.fit_transform(X_trans), columns=X.columns)

        # 3) DBSCAN
        eps = config.get('dbscan_eps', 0.02)
        factor = config.get('min_samples_factor', 0.05)
        min_samp = max(1, int(len(X_trans)*factor))

        db = DBSCAN(eps=eps, min_samples=min_samp)
        clusters = db.fit_predict(X_trans)
        zone_df['cluster_id'] = clusters

        # Filtrar outliers
        valid = zone_df[zone_df['cluster_id'] != -1]
        if valid.empty:
            log_event("WARNING", f"No clusters válidos en {zone_type}", module)
            return pd.DataFrame()

        # 4) Agrupar
        merged_rows = []
        for cid, grp in valid.groupby('cluster_id'):
            z_low  = grp[zone_low].min()
            z_up   = grp[zone_up].max()
            pivot_mean_price = grp[pivot_col].mean()
            last_ts = grp['timestamp'].max()
            zone_age = 0.0
            if pd.notnull(last_ts):
                zone_age = (valid['timestamp'].max() - last_ts).total_seconds()/3600.0

            merged_rows.append({
                'cluster_id': cid,
                'pivot_mean_price': pivot_mean_price,
                'zone_lower': z_low,
                'zone_upper': z_up,
                'last_pivot_ts': last_ts,
                'zone_freshness': zone_age
            })

        merged_df = pd.DataFrame(merged_rows)
        log_event("INFO", f"cluster_zones({zone_type}): total clusters => {merged_df['cluster_id'].nunique()}", module)
        return merged_df

    except Exception as e:
        log_event("ERROR", f"Error en cluster_zones({zone_type}): {str(e)}", module)
        return pd.DataFrame()


def zone_strength_ranking(df_zones: pd.DataFrame, df_main: pd.DataFrame, config: dict, zone_type:str) -> pd.DataFrame:
    """
    Combina repetición, volumen, freshness en un ranking no lineal.
    Opcionalmente, check fib_0.382, volume_delta si config indica. 
    Devuelve df_zones con column zone_strength.
    """
    module = "zones"
    if df_zones.empty:
        return df_zones

    try:
        # 1) Repeticiones
        # Por ejemplo, counting cuántas velas cayeron dentro de [zone_lower, zone_upper]
        # (placeholder approach)
        window = config.get('repetition_window', 90)
        df_zones['repetitions'] = df_zones.apply(
            lambda row: df_main[
                (df_main['close'] >= row['zone_lower']) &
                (df_main['close'] <= row['zone_upper'])
            ].shape[0],
            axis=1
        )

        # 2) Volumen Score
        # placeholder: could sum volume?
        df_zones['volume_score'] = 0.0
        # or from config if you want to gather e.g. df_main['volume'].mean() 
        # ...
        
        # 3) fib / volume_delta approach
        use_fib = config.get('use_fib', True)
        fib_factor = 0.0
        if use_fib and 'fib_0.382' in df_main.columns:
            fib_factor = df_main['fib_0.382'].mean()

        use_vol_delta = config.get('use_volume_delta', True)
        vol_delta_score = 0.0
        if use_vol_delta and 'volume_delta' in df_main.columns:
            vol_delta_score = df_main['volume_delta'].mean()

        # 4) Ranking
        for metric in ['repetitions', 'volume_score', 'zone_freshness']:
            col_rank = f'{metric}_rank'
            df_zones[col_rank] = df_zones[metric].rank(method='max', pct=True)

        # 5) Combine
        # e.g. strength_weights: [0.4, 0.3, 0.2, 0.1] 
        default_weights = [0.4, 0.3, 0.2, 0.1]  # reps, volume, fib, freshness?
        weights = config.get('strength_weights', default_weights)
        # We'll do a simple approach
        # zone_strength = sum( weight_i * rank_i ) ...
        # unify fib_factor, vol_delta_score as well ?

        zone_strength = (
            df_zones['repetitions_rank'] * weights[0] +
            df_zones['volume_score_rank'] * weights[1] +
            df_zones['zone_freshness'].rank(pct=True) * weights[2] +  # fresh rank
            fib_factor * (weights[3]*0.01) +  # placeholder factor
            vol_delta_score * (weights[3]*0.01) # or we do a separate
        )
        # we can refine or keep the approach

        # normalizar en [0..1]
        min_zs = zone_strength.min()
        max_zs = zone_strength.max() if zone_strength.max() != min_zs else (min_zs+1e-6)
        zone_strength = (zone_strength - min_zs) / (max_zs - min_zs)

        df_zones['zone_strength'] = zone_strength

        # 6) Logging
        log_feature_stats(df_zones, ['zone_strength','repetitions','volume_score','zone_freshness'], module)
        return df_zones

    except Exception as e:
        log_event("ERROR", f"Error en zone_strength_ranking: {str(e)}", module)
        return df_zones


def run_zones_pipeline(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline unificado:
     1) detect_pivots_adaptive => df con demand_zone_lower/upper, supply_zone_lower/upper
     2) cluster_zones(demand) + cluster_zones(supply) => unify
     3) zone_strength_ranking => ranking no lineal con placeholders
     4) concat => df_zones final con zone_strength, zone_type
     5) Retorna => (df_main, df_zones)
        df_main => 1 fila/vela con demand/supply col
        df_zones => 1 fila/cluster
    """
    module = "zones"
    log_event("INFO", "Iniciando run_zones_pipeline() unificado...", module)

    try:
        # 1) Detect Pivots
        df_main = detect_pivots_adaptive(df, config)

        # 2) cluster DEMAND
        dem_df = cluster_zones(df_main, 'demand', config)
        if not dem_df.empty:
            dem_df['zone_type'] = 'demand'
        # 3) cluster SUPPLY
        sup_df = cluster_zones(df_main, 'supply', config)
        if not sup_df.empty:
            sup_df['zone_type'] = 'supply'

        # Unir
        df_zones = pd.concat([dem_df, sup_df], ignore_index=True)
        if df_zones.empty:
            log_event("WARNING", "Ninguna zona clusterizada", module)
            return df_main, df_zones

        # 4) zone_strength
        df_zones = zone_strength_ranking(df_zones, df_main, config, zone_type='both')

        # Log
        log_event("INFO", f"Zonas detectadas total: {len(df_zones)}", module)
        mean_str = df_zones['zone_strength'].mean() if not df_zones.empty else 0
        log_event("INFO", f"Mean zone_strength = {mean_str:.3f}", module)

        log_event("INFO", "run_zones_pipeline completado con éxito.", module)
        return df_main, df_zones

    except Exception as e:
        log_event("ERROR", f"Error global en run_zones_pipeline: {str(e)}", module)
        return df, pd.DataFrame()
