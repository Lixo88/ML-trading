# prepare_data/indicators.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from prepare_data.logs import log_event, log_feature_stats

def calculate_technical_indicators(df: pd.DataFrame, timeframe: str, config: dict) -> pd.DataFrame:
    """
    Calcula indicadores técnicos avanzados con:
    - Prevención de data leakage (shift)
    - Normalización entre activos
    - Market Phase basado en ADX
    - Volumen inteligente
    """
    module_name = "indicators"
    log_event("INFO", f"Iniciando cálculo para {timeframe}", module_name)
    
    try:
        # =====================================================================
        # 1. Configuración Dinámica desde YAML
        # =====================================================================
        timeframe_cfg = config['indicators']['timeframes'].get(timeframe, {})
        global_cfg = config['indicators']
        
        # Parámetros
        sma_periods = timeframe_cfg.get('sma', [20, 50])
        fib_window = timeframe_cfg.get('fib_window', 30)
        adx_threshold = timeframe_cfg.get('adx_threshold', 25)
        volatility_window = timeframe_cfg.get('volatility_window', 14)
        scaling_factor = global_cfg['scaling_factors'].get(timeframe, 252)
        
        # =====================================================================
        # 2. Preprocesamiento Base (sin leakage)
        # =====================================================================
        df_clean = df.copy()
        
        # Returns usando shift(2) para prevenir leakage
        # (retorno de hoy se calcula con close de ayer)
        df_clean['returns'] = df_clean['close'].pct_change().shift(2)
        
        # =====================================================================
        # 3. Indicadores con Shift (T-1)
        # =====================================================================
        # Usamos close desplazado como base
        close_t1 = df_clean['close'].shift(1)
        
        # RSI
        df_clean['rsi'] = ta.rsi(close_t1, length=14)
        
        # MACD (12,26,9)
        macd = ta.macd(close_t1, fast=12, slow=26, signal=9)
        macd.columns = ['macd_line', 'macd_signal', 'macd_hist']
        df_clean = pd.concat([df_clean, macd], axis=1)
        
        # =====================================================================
        # 4. Market Phase (ADX + DI)
        # =====================================================================
        adx = ta.adx(df_clean['high'], df_clean['low'], close_t1, length=14)
        df_clean['di_plus'] = adx['DMP_14']
        df_clean['di_minus'] = adx['DMN_14']
        df_clean['adx'] = adx['ADX_14']
        
        df_clean['market_phase'] = np.select(
            [
                (df_clean['adx'] > adx_threshold) & (df_clean['di_plus'] > df_clean['di_minus']),
                (df_clean['adx'] > adx_threshold) & (df_clean['di_minus'] > df_clean['di_plus'])
            ],
            [1, -1],
            default=0
        )
        
        # =====================================================================
        # 5. Volumen Inteligente (sin inf/nan)
        # =====================================================================
        # EMA de volumen compra/venta
        green_candles = (df_clean['close'] > df_clean['open']).astype(int)
        df_clean['buy_vol_ema'] = (df_clean['volume'] * green_candles).ewm(span=14).mean()
        df_clean['sell_vol_ema'] = (df_clean['volume'] * (1 - green_candles)).ewm(span=14).mean()
        
        # Ratio con protección contra división por cero
        df_clean['vol_ratio'] = np.where(
            df_clean['sell_vol_ema'] < 1e-6,
            df_clean['buy_vol_ema'] / 1e-6,
            df_clean['buy_vol_ema'] / df_clean['sell_vol_ema']
        )
        
        # =====================================================================
        # 6. Normalización de Precios
        # =====================================================================
        sma_200 = close_t1.rolling(200).mean()
        df_clean['close_norm'] = close_t1 / sma_200
        
        # =====================================================================
        # 7. Volatilidad Anualizada
        # =====================================================================
        returns_vol = df_clean['returns'].rolling(volatility_window).std()
        df_clean['volatility'] = returns_vol * np.sqrt(scaling_factor)
        
        # =====================================================================
        # 8. Manejo Final de NaNs
        # =====================================================================
        # Eliminar filas con NaNs en columnas críticas
        critical_cols = ['rsi', 'macd_line', 'vol_ratio']
        df_clean.dropna(subset=critical_cols, inplace=True)
        
        # =====================================================================
        # 9. Logging de Calidad
        # =====================================================================
        log_feature_stats(df_clean, [
            'rsi', 'macd_line', 'vol_ratio', 
            'market_phase', 'volatility'
        ], module_name)
        
        if df_clean.empty:
            log_event("ERROR", "DataFrame vacío tras limpieza", module_name)
            return None
            
        log_event("INFO", f"Indicadores calculados. Filas: {len(df_clean)}", module_name)
        return df_clean

    except Exception as e:
        log_event("ERROR", f"Error crítico: {str(e)}", module_name)
        return None