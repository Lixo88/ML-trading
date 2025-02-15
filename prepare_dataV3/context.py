"""
context.py

Objetivos:
 1) Validar zonas entre temporalidades (ej. 4H vs 1D) usando Interval Trees.
 2) Integrar key_events (placeholder).
 3) Calcular un perfil de liquidez (opcional) para df_main.
 4) Calcular la relevancia final de cada zona (zone_relevance).
 5) Asignar la zona más fuerte (por zone_type) a cada vela en df_main:
    - closest_demand_relevance
    - closest_supply_relevance
 6) Retornar (df_main_context, df_zones_context).
"""

import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree

from prepare_data.logs import log_event, log_feature_stats

# ---------------------------------------------------------------------
# 1) VALIDACIÓN MULTI-TF con Interval Tree
# ---------------------------------------------------------------------
def validate_cross_temporal_zones(df_zones_lower: pd.DataFrame,
                                  df_zones_higher: pd.DataFrame,
                                  config: dict) -> pd.DataFrame:
    """
    Valida df_zones_lower vs df_zones_higher usando IntervalTree. 
    Marca 'validated_by_higher' = True si se solapan.
    
    Config:
      overlap_tolerance => [0.02 default] => expand zone_lower/upper un ± tol si deseas
    """
    module = "context"
    log_event("INFO", "Iniciando validación multi-TF (IntervalTree)", module)

    if df_zones_lower.empty or df_zones_higher.empty:
        log_event("INFO", "Uno de los DF está vacío. Sin validación multiTF.", module)
        df_zones_lower['validated_by_higher'] = False
        return df_zones_lower

    try:
        tol = config.get('overlap_tolerance', 0.02)
        
        # 1) Construir IntervalTree de la TF mayor
        higher_tree = IntervalTree()
        for _, row in df_zones_higher.iterrows():
            # expand zone? row['zone_lower']*(1-tol)? 
            z_low = row['zone_lower']*(1 - tol)
            z_up  = row['zone_upper']*(1 + tol)
            higher_tree.add(Interval(z_low, z_up, row.get('zone_strength', 1.0)))
        
        # 2) check overlap
        def check_overlap(r):
            expanded_lower = r['zone_lower']*(1 - tol)
            expanded_upper = r['zone_upper']*(1 + tol)
            matches = higher_tree.overlap(expanded_lower, expanded_upper)
            return len(matches) > 0

        df_zones_lower['validated_by_higher'] = df_zones_lower.apply(check_overlap, axis=1)

        val_sum = df_zones_lower['validated_by_higher'].sum()
        val_rate= val_sum / len(df_zones_lower)
        log_event("INFO", f"Zonas validadas: {val_sum}/{len(df_zones_lower)} ({val_rate:.1%})", module)
        return df_zones_lower
    except Exception as e:
        log_event("ERROR", f"Error en validate_cross_temporal_zones: {str(e)}", module)
        df_zones_lower['validated_by_higher'] = False
        return df_zones_lower


# ---------------------------------------------------------------------
# 2) Key Events Placeholder
# ---------------------------------------------------------------------
def integrate_key_events(df_zones: pd.DataFrame, key_events, config: dict) -> pd.DataFrame:
    """
    Asigna event_impact a df_zones. 
    Placeholder: event_impact=0 si no hay key_events.
    """
    module = "context"
    use_key = config.get('use_key_events', False)

    if not use_key or not key_events:
        log_event("INFO", "Key events desactivados => event_impact=0", module)
        df_zones['event_impact'] = 0.0
        return df_zones
    
    log_event("INFO", f"Integrando {len(key_events)} key_events...", module)
    # Approach: decaimiento, cross join con last_pivot_ts, etc.
    # Placeholder => event_impact=0
    df_zones['event_impact'] = 0.0

    return df_zones

# ---------------------------------------------------------------------
# 3) Liquidity Profile (Opcional)
# ---------------------------------------------------------------------
def calculate_liquidity_profile(df_main: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Crea col 'liquidity_profile' en df_main, si config lo indica.
    Window rolling, normalizado por ATR, etc.
    """
    module = "context"
    if df_main.empty:
        return df_main
    
    use_liquidity = config.get('use_liquidity_profile', False)
    if not use_liquidity:
        log_event("INFO", "Liquidez desactivada => skip", module)
        df_main['liquidity_profile'] = np.nan
        return df_main

    try:
        window_size    = config.get('liquidity_window', 20)
        price_tol      = config.get('price_tolerance', 0.02) # ±2%
        if 'ATR' not in df_main.columns:
            log_event("WARNING", "No ATR in df_main => liquidez approx", module)
            df_main['ATR'] = 1.0
        
        # rolling apply
        def calc_liquidity(subdf):
            mid_price = subdf['close'].iloc[-1]
            lower = mid_price*(1 - price_tol)
            upper = mid_price*(1 + price_tol)
            # sum volume in that window where close is in [lower,upper]
            mask = (subdf['close'] >= lower) & (subdf['close'] <= upper)
            return subdf.loc[mask, 'volume'].sum()

        df_main['liquidity_profile'] = df_main.rolling(window=window_size).apply(
            calc_liquidity, raw=False
        )
        df_main['liquidity_profile'] /= (df_main['ATR']+1e-9)

        return df_main
    except Exception as e:
        log_event("ERROR", f"calculate_liquidity_profile error: {str(e)}", module)
        df_main['liquidity_profile'] = np.nan
        return df_main

# ---------------------------------------------------------------------
# 4) Calcular Relevancia
# ---------------------------------------------------------------------
def calculate_zone_relevance(df_zones: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Combina zone_strength + validated_by_higher + event_impact + liquidez (opcional).
    zone_relevance = a*zone_strength + b*validated_by_higher + c*event_impact + d*liquidity???
    Colocar placeholders para submodel
    """
    module = "context"
    if df_zones.empty:
        df_zones['zone_relevance'] = []
        return df_zones
    
    try:
        a = config.get('strength_weight', 1.0)
        b = config.get('validation_weight', 1.0)
        c = config.get('event_impact_weight', 0.3)
        # d = config.get('liquidity_weight', 0.0) # si quisieras factor liquidez en zones

        if 'zone_strength' not in df_zones.columns:
            df_zones['zone_strength'] = 0.0
        if 'validated_by_higher' not in df_zones.columns:
            df_zones['validated_by_higher'] = False
        if 'event_impact' not in df_zones.columns:
            df_zones['event_impact'] = 0.0

        df_zones['zone_relevance'] = (
            a * df_zones['zone_strength'] +
            b * df_zones['validated_by_higher'].astype(float) +
            c * df_zones['event_impact']
        )

        return df_zones
    except Exception as e:
        log_event("ERROR", f"calculate_zone_relevance error: {str(e)}", module)
        df_zones['zone_relevance'] = 0.0
        return df_zones


# ---------------------------------------------------------------------
# 5) Asignar Zonas a velas
# ---------------------------------------------------------------------
def assign_zones_to_candles(df_main: pd.DataFrame, df_zones: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Usa IntervalTree para encontrar la zona con mayor zone_relevance 
    para demanda y oferta, generando:
     - df_main['closest_demand_relevance']
     - df_main['closest_supply_relevance']
    """
    module = "context"
    if df_main.empty or df_zones.empty:
        log_event("INFO", "assign_zones_to_candles => DF vacío, skip", module)
        return df_main
    
    # 1) Construir IntervalTrees para demand y supply
    demand_tree = IntervalTree()
    supply_tree = IntervalTree()

    for _, row in df_zones.iterrows():
        zone_begin = row['zone_lower']
        zone_end   = row['zone_upper']
        z_rel      = row.get('zone_relevance', 0.0)
        z_type     = row.get('zone_type', 'none')
        # Insert en el tree
        if z_type=='demand':
            demand_tree.add(Interval(zone_begin, zone_end, z_rel))
        elif z_type=='supply':
            supply_tree.add(Interval(zone_begin, zone_end, z_rel))

    def find_best_zone(tree, low_price, high_price):
        matches = tree.overlap(low_price, high_price)
        if not matches:
            return np.nan
        # pick the max zone_relevance 
        best = max(matches, key=lambda x: x.data)  # x.data => zone_relevance
        return best.data

    # 2) Para cada vela
    demand_vals = []
    supply_vals = []
    for _, row in df_main.iterrows():
        # Podrías ver open/close or low/high
        best_dem = find_best_zone(demand_tree, row['low'], row['high'])
        best_sup = find_best_zone(supply_tree, row['low'], row['high'])
        demand_vals.append(best_dem)
        supply_vals.append(best_sup)

    df_main['closest_demand_relevance'] = demand_vals
    df_main['closest_supply_relevance'] = supply_vals

    coverage_dem = df_main['closest_demand_relevance'].notna().mean()
    coverage_sup = df_main['closest_supply_relevance'].notna().mean()
    log_event("INFO", 
        f"Asig. zona: coverage demand={coverage_dem:.1%}, supply={coverage_sup:.1%}",
        module
    )

    return df_main


# ---------------------------------------------------------------------
# 6) Pipeline Principal
# ---------------------------------------------------------------------
def run_context_pipeline(df_main: pd.DataFrame,
                         df_zones: pd.DataFrame,
                         df_zones_higher: pd.DataFrame=None,
                         key_events=None,
                         config: dict=None):
    """
    1) Liquidity Profile (opcional)
    2) Validar multiTF => validated_by_higher
    3) integrate_key_events => event_impact
    4) calculate_zone_relevance => zone_relevance
    5) assign_zones_to_candles => df_main con closest_demand_relevance, supply
    Retorna (df_main_context, df_zones_context)
    """
    module = "context"
    log_event("INFO", "Iniciando run_context_pipeline unify", module)

    if config is None:
        config = {}

    try:
        # 1) Liquidity
        df_main_ctx = calculate_liquidity_profile(df_main, config)

        # 2) Validación multiTF
        multi_tf = config.get('multi_tf_validation', False)
        df_zones_ctx = df_zones.copy()

        if multi_tf and df_zones_higher is not None and not df_zones_higher.empty:
            df_zones_ctx = validate_cross_temporal_zones(df_zones_ctx, df_zones_higher, config)
        else:
            df_zones_ctx['validated_by_higher'] = False
        
        # 3) Key events => event_impact
        df_zones_ctx = integrate_key_events(df_zones_ctx, key_events, config)

        # 4) zone_relevance
        df_zones_ctx = calculate_zone_relevance(df_zones_ctx, config)

        # 5) Asignar Zonas => demand / supply 
        assign_bool = config.get('assign_zone_to_candles', False)
        if assign_bool:
            df_main_ctx = assign_zones_to_candles(df_main_ctx, df_zones_ctx, config)

        log_event("INFO", "run_context_pipeline completado", module)
        return df_main_ctx, df_zones_ctx

    except Exception as e:
        log_event("ERROR", f"run_context_pipeline error: {str(e)}", module)
        return df_main, df_zones
