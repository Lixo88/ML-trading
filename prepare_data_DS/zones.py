# prepare_data/zones.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from prepare_data.logs import log_event, log_feature_stats

def detect_pivots(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Detección adaptativa de pivots usando ATR dinámico y filtros de volumen."""
    module = "zones"
    log_event("INFO", "Iniciando detección de pivots", module)
    
    try:
        # =====================================================================
        # 1. Configuración Dinámica
        # =====================================================================
        atr_multiplier = config.get('atr_multiplier', 1.5)
        min_pivot_window = config.get('pivot_window', 3)
        volume_z_threshold = config.get('volume_zscore_threshold', 2.0)
        
        # =====================================================================
        # 2. Ventana Adaptativa Basada en Volatilidad (ATR)
        # =====================================================================
        avg_atr = df['ATR'].rolling(100).mean().iloc[-1]
        dynamic_window = np.clip(
            int(np.ceil(5 * (df['ATR'].iloc[-1] / avg_atr))),  # Corregido paréntesis
            min_pivot_window, 
            10
        )
        
        # =====================================================================
        # 3. Detección de Swing Lows/Highs
        # =====================================================================
        df['swing_low'] = (
            df['low']
            .rolling(window=dynamic_window*2+1, center=True)
            .min()
        )
        df['is_swing_low'] = (df['low'] == df['swing_low']) & (
            df.index > dynamic_window
        )

        df['swing_high'] = (
            df['high']
            .rolling(window=dynamic_window*2+1, center=True)
            .max()
        )
        df['is_swing_high'] = (df['high'] == df['swing_high']) & (
            df.index > dynamic_window
        )

        # =====================================================================
        # 4. Filtrado por Volumen Significativo (Z-Score)
        # =====================================================================
        volume_mean = df['volume'].rolling(50).mean()
        volume_std = df['volume'].rolling(50).std().replace(0, 1e-6)
        df['volume_zscore'] = (df['volume'] - volume_mean) / volume_std
        
        demand_mask = (
            df['is_swing_low'] 
            & (df['volume_zscore'] >= volume_z_threshold)
            & (df['close'] > df['open'])  # Velas alcistas
        )
        
        supply_mask = (
            df['is_swing_high'] 
            & (df['volume_zscore'] >= volume_z_threshold)
            & (df['close'] < df['open'])  # Velas bajistas
        )

        # =====================================================================
        # 5. Creación de Zonas Base
        # =====================================================================
        df_pivots = pd.DataFrame()
        df_pivots['type'] = np.nan
        df_pivots['price'] = np.nan
        
        df_pivots.loc[demand_mask, 'type'] = 'demand'
        df_pivots.loc[demand_mask, 'price'] = df.loc[demand_mask, 'low'] - df.loc[demand_mask, 'ATR'] * atr_multiplier
        
        df_pivots.loc[supply_mask, 'type'] = 'supply'
        df_pivots.loc[supply_mask, 'price'] = df.loc[supply_mask, 'high'] + df.loc[supply_mask, 'ATR'] * atr_multiplier
        
        df_pivots.dropna(inplace=True)
        
        log_event("INFO", f"Pivots detectados: {len(df_pivots)} (Demanda: {demand_mask.sum()}, Oferta: {supply_mask.sum()})", module)
        return df_pivots

    except Exception as e:
        log_event("ERROR", f"Error en detección de pivots: {str(e)}", module)
        return pd.DataFrame()

def cluster_zones(pivots: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clustering no lineal con normalización avanzada."""
    module = "zones"
    
    if pivots.empty:
        log_event("WARNING", "No hay pivots para clusterizar", module)
        return pd.DataFrame()

    try:
        # =====================================================================
        # 1. Ingeniería de Features para Clustering
        # =====================================================================
        X = pd.DataFrame()
        X['price'] = np.log(pivots['price'])  # Precio en escala logarítmica
        X['volume'] = QuantileTransformer().fit_transform(pivots[['volume_zscore']])
        X['time'] = (pivots.index - pivots.index.min()).total_seconds() / 3600  # Horas desde el primer pivot
        
        # =====================================================================
        # 2. Normalización No Lineal
        # =====================================================================
        transformer = PowerTransformer()
        X_trans = transformer.fit_transform(X)
        
        # =====================================================================
        # 3. DBSCAN Adaptativo
        # =====================================================================
        eps = config.get('dbscan_eps', 0.02)
        min_samples = max(1, int(len(X) * 0.05))  # Mínimo 5% de los puntos
        
        db = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = db.fit_predict(X_trans)
        
        # =====================================================================
        # 4. Procesamiento de Clusters
        # =====================================================================
        pivots['cluster'] = clusters
        valid_clusters = pivots[pivots['cluster'] != -1]
        
        # Agregación por cluster
        df_zones = valid_clusters.groupby(['cluster', 'type']).agg({
            'price': ['min', 'max', 'mean'],
            'volume_zscore': 'sum',
            'time': 'max'
        }).reset_index()
        
        df_zones.columns = ['cluster', 'type', 'zone_lower', 'zone_upper', 'price_mean', 'volume_score', 'last_active']
        
        # =====================================================================
        # 5. Cálculo de Freshness Dinámico
        # =====================================================================
        max_time = df_zones['last_active'].max()
        df_zones['freshness'] = 1 - (max_time - df_zones['last_active']) / (24 * 7)  # Decaimiento semanal
        
        log_event("INFO", f"Zonas clusterizadas: {len(df_zones)} (Clusters válidos: {len(df_zones['cluster'].unique())})", module)
        return df_zones

    except Exception as e:
        log_event("ERROR", f"Error en clustering: {str(e)}", module)
        return pd.DataFrame()

def calculate_zone_strength(zones: pd.DataFrame, df_main: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cálculo de fuerza de zona con interacciones no lineales."""
    module = "zones"
    
    if zones.empty:
        return zones
    
    try:
        # =====================================================================
        # 1. Cálculo de Repeticiones Históricas
        # =====================================================================
        window = config.get('repetition_window', 90)
        zones['repetitions'] = zones.apply(
            lambda row: df_main[
                (df_main['close'] >= row['zone_lower']) & 
                (df_main['close'] <= row['zone_upper'])
            ].shape[0], 
            axis=1
        )
        
        # =====================================================================
        # 2. Ranking Percentil No Lineal
        # =====================================================================
        for metric in ['repetitions', 'volume_score', 'freshness']:
            zones[f'{metric}_rank'] = zones[metric].rank(pct=True, method='max')
        
        # =====================================================================
        # 3. Combinación Ponderada (Placeholder para modelo ML)
        # =====================================================================
        weights = np.array(config.get('strength_weights', [0.5, 0.3, 0.2]))  # reps, volume, freshness
        ranks = zones[['repetitions_rank', 'volume_score_rank', 'freshness_rank']].values
        zones['strength'] = np.dot(ranks, weights)
        
        # =====================================================================
        # 4. Normalización Final
        # =====================================================================
        zones['strength'] = (zones['strength'] - zones['strength'].min()) / \
                           (zones['strength'].max() - zones['strength'].min())
        
        log_feature_stats(zones, ['strength', 'repetitions', 'volume_score'], module)
        return zones

    except Exception as e:
        log_event("ERROR", f"Error en cálculo de fuerza: {str(e)}", module)
        return zones

def run_zones_pipeline(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de detección de zonas."""
    module = "zones"
    
    try:
        # Paso 1: Detección de Pivotes
        df_pivots = detect_pivots(df, config)
        
        # Paso 2: Clustering
        df_zones = cluster_zones(df_pivots, config)
        
        # Paso 3: Cálculo de Fuerza
        df_zones = calculate_zone_strength(df_zones, df, config)
        
        # Paso 4: Merge con datos originales
        df_main = df.join(df_pivots[['type', 'price']], how='left')
        
        log_event("INFO", f"Pipeline completado. Zonas detectadas: {len(df_zones)}", module)
        return df_main, df_zones
    
    except Exception as e:
        log_event("ERROR", f"Error en pipeline: {str(e)}", module)
        return pd.DataFrame(), pd.DataFrame()