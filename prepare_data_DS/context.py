# prepare_data/context.py
from intervaltree import Interval, IntervalTree
import pandas as pd
import numpy as np
from prepare_data.logs import log_event, log_feature_stats

def validate_cross_temporal_zones(df_zones_current: pd.DataFrame, 
                                 df_zones_higher: pd.DataFrame,
                                 config: dict) -> pd.DataFrame:
    """
    Valida zonas contra temporalidad superior usando Interval Trees.
    Devuelve DataFrame con columna 'validated_by_higher'.
    """
    module = "context"
    log_event("INFO", "Iniciando validación multi-TF", module)
    
    try:
        # 1. Construir Interval Tree de la temporalidad superior
        higher_tree = IntervalTree()
        for _, zone in df_zones_higher.iterrows():
            higher_tree.add(Interval(zone['zone_lower'], zone['zone_upper'], zone['strength']))
        
        # 2. Búsqueda eficiente de overlaps
        def check_overlap(row):
            return len(higher_tree.overlap(row['zone_lower'], row['zone_upper'])) > 0
        
        df_zones_current['validated_by_higher'] = df_zones_current.apply(check_overlap, axis=1)
        
        # 3. Métricas de validación
        val_rate = df_zones_current['validated_by_higher'].mean()
        log_event("INFO", 
                 f"Zonas validadas: {df_zones_current['validated_by_higher'].sum()}/"
                 f"{len(df_zones_current)} ({val_rate:.1%})",
                 module)
        
        return df_zones_current
    
    except Exception as e:
        log_event("ERROR", f"Validación multi-TF fallida: {str(e)}", module)
        df_zones_current['validated_by_higher'] = False
        return df_zones_current

def calculate_liquidity_profile(df_main: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calcula perfil de liquidez en ventana dinámica alrededor del precio."""
    module = "context"
    
    try:
        window_size = config.get('liquidity_window', 20)
        price_tolerance = config.get('price_tolerance', 0.02)  # ±2%
        
        df = df_main.copy()
        df['price_window_lower'] = df['close'] * (1 - price_tolerance)
        df['price_window_upper'] = df['close'] * (1 + price_tolerance)
        
        # 1. Liquidez histórica en ventana de precios
        df['liquidity_profile'] = df.rolling(window=window_size).apply(
            lambda x: x['volume'][(x['close'] >= x['price_window_lower']) & 
                                (x['close'] <= x['price_window_upper'])].sum(),
            raw=False
        )
        
        # 2. Normalizar por volatilidad
        df['liquidity_profile'] /= df['ATR'] + 1e-6  # Evitar división por cero
        
        log_feature_stats(df, ['liquidity_profile'], module)
        return df
    
    except Exception as e:
        log_event("ERROR", f"Cálculo de liquidez fallido: {str(e)}", module)
        df_main['liquidity_profile'] = np.nan
        return df_main

def assign_zones_to_candles(df_main: pd.DataFrame, 
                           df_zones: pd.DataFrame,
                           config: dict) -> pd.DataFrame:
    """Asigna la zona más relevante a cada vela usando búsqueda espacial."""
    module = "context"
    
    try:
        # 1. Construir Interval Tree optimizado
        zone_tree = IntervalTree()
        for _, zone in df_zones.iterrows():
            zone_tree.add(Interval(zone['zone_lower'], 
                           zone['zone_upper'], 
                           (zone['strength'], zone['type'])))
        
        # 2. Búsqueda por vela
        def get_zone_info(row):
            matches = zone_tree.overlap(row['low'], row['high'])
            if not matches:
                return (np.nan, np.nan, np.nan)
            
            best_zone = max(matches, key=lambda x: x.data[0])
            return (best_zone.data[0], best_zone.begin, best_zone.end)
        
        # 3. Aplicación vectorizada
        results = np.array([get_zone_info(row) for _, row in df_main.iterrows()])
        
        # 4. Asignar al DataFrame
        df_main['zone_strength'] = results[:, 0]
        df_main['zone_lower'] = results[:, 1]
        df_main['zone_upper'] = results[:, 2]
        
        # 5. Estadísticas
        coverage = df_main['zone_strength'].notna().mean()
        log_event("INFO", 
                 f"Cobertura de zonas: {coverage:.1%} ({df_main['zone_strength'].notna().sum()}/"
                 f"{len(df_main)})",
                 module)
        
        return df_main
    
    except Exception as e:
        log_event("ERROR", f"Asignación de zonas fallida: {str(e)}", module)
        return df_main

def run_context_pipeline(df_main: pd.DataFrame,
                        df_zones: pd.DataFrame,
                        df_zones_higher: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """Orquesta todo el procesamiento de contexto."""
    module = "context"
    log_event("INFO", "Iniciando pipeline de contexto", module)
    
    try:
        # 1. Validación Multi-TF
        df_zones_val = validate_cross_temporal_zones(df_zones, df_zones_higher, config)
        
        # 2. Cálculo de Relevancia (CORREGIDO)
        strength_weight = config.get('strength_weight', 0.7)
        validation_weight = config.get('validation_weight', 0.3)
        
        df_zones_val['zone_relevance'] = (
            strength_weight * df_zones_val['strength'] +
            validation_weight * df_zones_val['validated_by_higher'].astype(float)
        )
        
        # 3. Perfil de Liquidez
        df_main = calculate_liquidity_profile(df_main, config)
        
        # 4. Asignación de Zonas (ACTUALIZADO para usar zone_relevance)
        df_main = assign_zones_to_candles(df_main, df_zones_val, config)
        
        # 5. Unir DataFrames (CON NOMBRE CORRECTO)
        df_final = pd.merge(
            df_main, 
            df_zones_val[['zone_lower', 'zone_upper', 'zone_relevance', 'type']],
            how='left', 
            on=['zone_lower', 'zone_upper']
        )
        
        log_event("INFO", f"Pipeline completado. Columnas finales: {list(df_final.columns)}", module)
        return df_final
    
    except Exception as e:
        log_event("ERROR", f"Error en pipeline de contexto: {str(e)}", module)
        return pd.DataFrame()