"""
indicators.py

Unificación de la lógica para:
 - Prevención de Leakage (shift(2) en returns, shift(1) en RSI/MACD/etc.)
 - Cálculo de Bollinger, ATR, ADX, Fibonacci, SMAs, Delta Volumen
 - Market Phase con ADX threshold
 - Volumen Inteligente (EWMA buy/sell)
 - Normalización de precios con SMA 200
 - Body/Wick analysis (si se desea conservar en el DF)
 - Logging detallado y placeholders para submodelo

Las columnas clave generadas:
 - returns (shift(2))
 - rsi, macd_line, macd_signal, macd_hist, atr, adx, di_plus, di_minus
 - bb_upper, bb_lower, bb_ratio
 - fib_0.236 ... fib_0.786, dist_to_fib
 - buy_volume, sell_volume, volume_delta, volume_ratio, buy_vol_ema, sell_vol_ema
 - market_phase (basado en adx_threshold)
 - close_norm (precio normalizado con SMA 200)
 - sma_X (dinámicas)
 - volatility (rolling std * sqrt(scaling_factor))
 - body, lower_wick, upper_wick (placeholder para otras etapas)
 
Nota: SHIFT(2) en returns significa que
 returns[t] = (close[t-1] - close[t-2]) / close[t-2]
 y SHIFT(1) en RSI/MACD/ADX implica que sus valores en la vela t 
 solo ven precios hasta la vela t-1. 
Esto reduce la posibilidad de leakage en ML.

Cualquier peso o threshold (e.g. adx_threshold=25) puede ser
 refinado con un submodelo XGBoost a futuro.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from prepare_data.logs import log_event, log_feature_stats

def calculate_technical_indicators(df: pd.DataFrame, timeframe: str, config: dict) -> pd.DataFrame:
    """
    Unifica lógica de indicadores con no-leakage, market phase, volumen inteligente,
    Bollinger, ATR, Fibonacci, SMAs, etc.

    Args:
        df (pd.DataFrame): DataFrame con al menos [timestamp, open, high, low, close, volume].
        timeframe (str): Temporalidad, p.ej. '1d', '4h', '1h', '15m'.
        config (dict): Sub-config en config['indicators']. Contiene:
            timeframes[timeframe]:
                - sma: [20, 50]           # SMAs
                - fib_window: 45
                - adx_threshold: 25
                - volatility_window: 14
                ...
            scaling_factors:
                "1d": 252, "4h":63, etc.
    
    Returns:
        pd.DataFrame: DataFrame con columnas de indicadores.
        Devuelve None si ocurre un error crítico o DF vacío tras limpieza.
    """
    module_name = "indicators"
    log_event("INFO", f"Iniciando cálculo de indicadores para {timeframe}", module_name)

    try:
        # ----------------------------------------------------------------------
        # 1) Configuración según timeframe
        # ----------------------------------------------------------------------
        tf_cfg = config.get('timeframes', {}).get(timeframe, {})
        sma_periods        = tf_cfg.get('sma', [20, 50])
        fib_window         = tf_cfg.get('fib_window', 30)
        volatility_window  = tf_cfg.get('volatility_window', 14)
        adx_threshold      = tf_cfg.get('adx_threshold', 25)

        # scaling factor
        scaling_factors = config.get('scaling_factors', {})
        scale = scaling_factors.get(timeframe, 252)

        # ----------------------------------------------------------------------
        # 2) Copia del DF y SHIFT(2) en returns
        # ----------------------------------------------------------------------
        df_clean = df.copy()

        # SHIFT(2) => returns[t] usa close[t-2] y close[t-1]
        # Avoid leakage: la vela t no ve el precio t
        df_clean['returns'] = df_clean['close'].pct_change().shift(2)

        # guardamos body y wicks si se usan en otras etapas
        df_clean['body'] = df_clean['close'] - df_clean['open']
        df_clean['total_range'] = (df_clean['high'] - df_clean['low']).replace(0, np.nan)
        df_clean['lower_wick'] = (df_clean['open'] - df_clean['low']).clip(lower=0)
        df_clean['upper_wick'] = (df_clean['high'] - df_clean['close']).clip(lower=0)

        # SHIFT(1) para RSI/MACD/ADX
        close_t1 = df_clean['close'].shift(1)
        high_t1  = df_clean['high'].shift(1)
        low_t1   = df_clean['low'].shift(1)

        # ----------------------------------------------------------------------
        # 3) RSI, MACD
        # ----------------------------------------------------------------------
        df_clean['rsi'] = ta.rsi(close_t1, length=14)

        macd = ta.macd(close_t1, fast=12, slow=26, signal=9)
        # renombrar => ['macd_line','macd_signal','macd_hist']
        # Ojo: por defecto, macd = [MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9]
        macd.columns = ['macd_line','macd_hist','macd_signal']
        df_clean = pd.concat([df_clean, macd], axis=1)

        # ----------------------------------------------------------------------
        # 4) ATR, ADX
        # ----------------------------------------------------------------------
        # ATR/ADX basados en shift(1) => high_t1, low_t1, close_t1
        # O se hace ATR en T real? Depende del approach. 
        # Para no leakage total, se hace en T-1.
        atr_series = ta.atr(high_t1, low_t1, close_t1, length=14)
        adx_df     = ta.adx(high_t1, low_t1, close_t1, length=14)
        # adx_df => [ADX_14, DMP_14, DMN_14]
        adx_df.columns = ['adx','di_plus','di_minus']

        df_clean['atr'] = atr_series
        df_clean[['adx','di_plus','di_minus']] = adx_df[['adx','di_plus','di_minus']]

        # Market Phase => adx_threshold
        df_clean['market_phase'] = np.select(
            [
                (df_clean['adx'] > adx_threshold) & (df_clean['di_plus'] > df_clean['di_minus']),
                (df_clean['adx'] > adx_threshold) & (df_clean['di_minus'] > df_clean['di_plus'])
            ],
            [1, -1],
            default=0
        )

        # ----------------------------------------------------------------------
        # 5) Bollinger Bands
        # ----------------------------------------------------------------------
        # En T-1 o T real? Aquí optamos T-1 para no leak. 
        # shift(1) => rolling(20) => no leakage. 
        rolling_close_t1 = close_t1.rolling(20)
        sma_20 = rolling_close_t1.mean()
        std_20 = rolling_close_t1.std()

        df_clean['bb_upper'] = sma_20 + 2 * std_20
        df_clean['bb_lower'] = sma_20 - 2 * std_20
        df_clean['bb_ratio'] = (close_t1 - df_clean['bb_lower']) / (df_clean['bb_upper'] - df_clean['bb_lower'])

        # ----------------------------------------------------------------------
        # 6) Fibonacci Rolling
        # ----------------------------------------------------------------------
        # Rolling max/min con fib_window en high_t1, low_t1
        max_price = high_t1.rolling(fib_window).max()
        min_price = low_t1.rolling(fib_window).min()
        total_range = max_price - min_price

        fib_levels = {
            'fib_0.236': max_price - total_range * 0.236,
            'fib_0.382': max_price - total_range * 0.382,
            'fib_0.5':   max_price - total_range * 0.5,
            'fib_0.618': max_price - total_range * 0.618,
            'fib_0.786': max_price - total_range * 0.786
        }
        for level, val in fib_levels.items():
            df_clean[level] = val
            df_clean[f'dist_to_{level}'] = (close_t1 - val) / (val.replace(0, np.nan)) * 100

        # ----------------------------------------------------------------------
        # 7) Volumen Inteligente
        # ----------------------------------------------------------------------
        green_candles = (df_clean['close'] > df_clean['open']).astype(int)
        df_clean['buy_volume']  = df_clean['volume'] * green_candles
        df_clean['sell_volume'] = df_clean['volume'] * (1 - green_candles)

        # EWM
        df_clean['buy_vol_ema']  = df_clean['buy_volume'].ewm(span=14).mean()
        df_clean['sell_vol_ema'] = df_clean['sell_volume'].ewm(span=14).mean()

        df_clean['volume_delta'] = df_clean['buy_volume'] - df_clean['sell_volume']

        df_clean['volume_ratio'] = np.where(
            df_clean['sell_vol_ema'] < 1e-9,
            df_clean['buy_vol_ema'] / 1e-9,
            df_clean['buy_vol_ema'] / df_clean['sell_vol_ema']
        )

        # ----------------------------------------------------------------------
        # 8) SMA dinámicas
        # ----------------------------------------------------------------------
        # Nota: si preferimos SHIFT(1) => rolling sobre close_t1
        # o si preferimos rolling actual => df_clean['close']
        # iremos con close_t1 para no leak
        for period in sma_periods:
            df_clean[f'sma_{period}'] = close_t1.rolling(period).mean()

        # ----------------------------------------------------------------------
        # 9) Normalización y Volatilidad
        # ----------------------------------------------------------------------
        # close_norm => close_t1 / sma_200 (shifted)
        sma_200 = close_t1.rolling(200).mean()
        df_clean['close_norm'] = close_t1 / sma_200

        # Volatilidad => rolling std( returns ) * sqrt(scale)
        returns_vol = df_clean['returns'].rolling(volatility_window).std()
        df_clean['volatility'] = returns_vol * np.sqrt(scale)

        # ----------------------------------------------------------------------
        # 10) Manejo de NaNs y Placeholder Submodelo
        # ----------------------------------------------------------------------
        # Drop filas con NaNs en columnas clave
        # Si muchas => quizá ser flexible. 
        critical_cols = ['rsi','macd_line','volume_ratio']
        df_clean.dropna(subset=critical_cols, inplace=True)

        # A futuro => placeholders p.ej. adx_threshold, ratio de EWM 
        # se podrían refinar con un submodelo XGBoost entrenado históricamente.

        # ----------------------------------------------------------------------
        # 11) Logs y Feature Stats
        # ----------------------------------------------------------------------
        # Revisar duplicados de columnas
        if df_clean.columns.duplicated().any():
            log_event("WARNING", "Columnas duplicadas detectadas tras merges", module_name)

        # Log stats
        features_to_log = [
            'rsi','macd_line','vol_ratio','market_phase','close_norm','volatility','adx'
        ]
        log_feature_stats(df_clean, features_to_log, module=module_name)

        if df_clean.empty:
            log_event("ERROR", "DataFrame vacío tras limpieza de NaNs", module_name)
            return None

        log_event("INFO", f"Indicadores calculados. Filas finales: {len(df_clean)}", module_name)
        return df_clean

    except Exception as e:
        log_event("ERROR", f"Error en {timeframe}: {str(e)}", module_name)
        return None
