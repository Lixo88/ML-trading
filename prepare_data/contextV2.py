"""
context.py

Script para:
 - Validar y ajustar zonas entre dos temporalidades (opcional).
 - Integrar eventos clave (key_events) y asignar event_impact.
 - Calcular zone_relevance combinando zone_raw_strength con el impacto de eventos y la validación multi-TF.

Retorna (df_main_context, df_zones_context) o, si no hay multi-TF, al menos
(df_main_context, df_zones_context) para la misma TF.
"""

import pandas as pd
import numpy as np
from prepare_data.logs import log_event

# Config por defecto
DEFAULT_CONTEXT_CONFIG = {
    'multi_tf_validation': True,  # si queremos validar 4H vs 1D
    'overlap_tolerance': 0.02,    # Porcentaje (o ATR-based) para considerar una zona validada
    'use_key_events': True,
    'decay_hours': 12,            # Ventana de ±12h para asignar impacto de un evento
    'relevance_model': None,      # Ruta a un sub-modelo real (XGBoost, etc.) o None -> placeholder
}

# ---------------------------------------------------------------------
# 1) VALIDAR ZONAS ENTRE TEMPORALIDADES
# ---------------------------------------------------------------------
def validate_cross_temporal_zones(df_zones_lower: pd.DataFrame,
                                  df_zones_higher: pd.DataFrame,
                                  config=None) -> pd.DataFrame:
    """
    Marca en df_zones_lower cuáles zonas se solapan con df_zones_higher.
    Por ejemplo, si 'zone_lower' < zone_upper_higher + tolerance, y viceversa.
    Añade col 'validated_by_higher' = True/False. 
    """
    if config is None:
        config = DEFAULT_CONTEXT_CONFIG

    tol = config['overlap_tolerance']  # 0.02 => ~2% de solapamiento

    # Nueva columna en df_zones_lower
    df_zones_lower['validated_by_higher'] = False

    # Dos opciones de cómo hacer la comparación:
    # A) Loop (ineficiente) 
    # B) merges condicionales vectorizados
    # C) approach simple: for each zone lower, check if any zone higher overlaps
    # A modo de ejemplo, un for + logs:

    for i, rowL in df_zones_lower.iterrows():
        # Filtrar df_zones_higher con similar zone_type
        same_type = df_zones_higher[df_zones_higher['zone_type'] == rowL['zone_type']]
        # Overlap cond:
        #   rowL.zone_lower <= rowH.zone_upper*(1+tol)  and rowL.zone_upper >= rowH.zone_lower*(1-tol)
        overlap_mask = (
            (same_type['zone_lower'] <= rowL['zone_upper']*(1+tol)) &
            (same_type['zone_upper'] >= rowL['zone_lower']*(1-tol))
        )
        if overlap_mask.any():
            df_zones_lower.loc[i, 'validated_by_higher'] = True

    return df_zones_lower

# ---------------------------------------------------------------------
# 2) INTEGRAR KEY EVENTS
# ---------------------------------------------------------------------
def integrate_key_events(df_zones: pd.DataFrame,
                         key_events: list,
                         config=None) -> pd.DataFrame:
    """
    - 'key_events' es una lista de dicts: [{'timestamp':..., 'impact':..., 'desc':...}, ...]
    - Asignamos event_impact en df_zones si su 'last_pivot_ts' (por ej.) está cerca (± config['decay_hours']).
    - Suma o promedia los impactos si varios eventos solapan.
    """
    if config is None:
        config = DEFAULT_CONTEXT_CONFIG

    if not config['use_key_events'] or not key_events:
        df_zones['event_impact'] = 0.0
        return df_zones

    decay_hrs = config['decay_hours']

    # Inicializar
    df_zones['event_impact'] = 0.0

    for evt in key_events:
        evt_time = evt['timestamp']
        evt_baseimpact = evt.get('impact', 1.0)  # un valor base
        evt_desc = evt.get('desc', '')

        # Mascara de "cercanía" en horas, usando 'last_pivot_ts' o similar
        if 'last_pivot_ts' in df_zones.columns:
            dt_hours = (df_zones['last_pivot_ts'] - evt_time).abs().dt.total_seconds()/3600
            # si dt_hours <= decay_hrs => sumamos
            near_mask = (dt_hours <= decay_hrs)
            # Ejemplo: decaimiento lineal
            decay_factor = np.maximum(0.0, 1 - (dt_hours / decay_hrs))
            # Ajustar
            additional_impact = evt_baseimpact * decay_factor
            # Donde near_mask, sumamos
            df_zones.loc[near_mask, 'event_impact'] += additional_impact[near_mask]
        else:
            # si no existe last_pivot_ts, no hacemos nada
            pass

    return df_zones


# ---------------------------------------------------------------------
# 3) CALCULAR RELEVANCIA FINAL (zone_relevance)
# ---------------------------------------------------------------------
def calculate_zone_relevance(df_zones: pd.DataFrame,
                             config=None) -> pd.DataFrame:
    """
    Combina zone_raw_strength con event_impact y validación multi-TF (if any).
    E.g. zone_relevance = zone_raw_strength + 0.3*event_impact + (validated_by_higher? 1.0 : 0)
    O usar un sub-modelo. 
    """
    if config is None:
        config = DEFAULT_CONTEXT_CONFIG

    if df_zones.empty:
        df_zones['zone_relevance'] = []
        return df_zones

    # Ejemplo simple:
    # zone_relevance = zone_raw_strength + 0.3*(event_impact) + 1*(validated_by_higher)
    # asumiendo: validated_by_higher => boolean => cast to 1 or 0
    if 'zone_raw_strength' not in df_zones.columns:
        df_zones['zone_raw_strength'] = 0.0

    if 'event_impact' not in df_zones.columns:
        df_zones['event_impact'] = 0.0

    if 'validated_by_higher' not in df_zones.columns:
        df_zones['validated_by_higher'] = False

    # Sub-model real
    # if config['relevance_model'] is not None:
    #     # Cargar pickle, features = ...
    #     pass
    # else:
    # Placeholder lineal
    validated_factor = df_zones['validated_by_higher'].astype(float)
    df_zones['zone_relevance'] = (
        df_zones['zone_raw_strength'] 
        + 0.3*df_zones['event_impact']
        + 1.0*validated_factor
    )

    return df_zones

# ---------------------------------------------------------------------
# 4) PIPELINE PRINCIPAL
# ---------------------------------------------------------------------
def run_context_pipeline(df_main: pd.DataFrame,
                         df_zones: pd.DataFrame,
                         # Para multiTF:
                         df_zones_higher: pd.DataFrame = None,
                         key_events: list = None,
                         config=None):
    """
    Orquesta la lógica 'context':
     1) Si df_zones_higher no es None => validate_cross_temporal_zones
     2) integrate_key_events -> event_impact
     3) calculate_zone_relevance -> zone_relevance
     4) Retorna (df_main_context, df_zones_context)
    """
    if config is None:
        config = DEFAULT_CONTEXT_CONFIG

    # Clonar df_main, df_zones si queremos no modificar los originales
    df_main_context = df_main.copy()
    df_zones_context = df_zones.copy()

    # 1) Validación multiTF, si df_zones_higher no es None
    if config['multi_tf_validation'] and df_zones_higher is not None and not df_zones_higher.empty:
        log_event("INFO", "Validando multi-TimeFrame con df_zones_higher...", module="context")
        df_zones_context = validate_cross_temporal_zones(df_zones_context, df_zones_higher, config)

    # 2) Integrar key events
    if key_events:
        log_event("INFO", f"Integrando {len(key_events)} key_events en df_zones...", module="context")
        df_zones_context = integrate_key_events(df_zones_context, key_events, config)
    else:
        df_zones_context['event_impact'] = 0.0

    # 3) Calcular zone_relevance
    log_event("INFO", "Calculando zone_relevance...", module="context")
    df_zones_context = calculate_zone_relevance(df_zones_context, config)

    # (Opcional) Podrías asignar la 'relevance' a df_main según la zona “más cercana” a cada vela
    #  => algo como 'closest_zone_relevance'
    # for debugging, etc. 
    # omitted here for brevity.

    log_event("INFO", "run_context_pipeline completado.", module="context")

    # 4) Retornamos
    return df_main_context, df_zones_context
