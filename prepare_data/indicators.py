import pandas as pd
import numpy as np
import pandas_ta as ta
from prepare_data.logs import log_event

def calculate_technical_indicators(df, timeframe):
    """
    Calcula indicadores técnicos con parámetros dinámicos, incluyendo:
    - RSI, MACD, ATR, ADX (usando pandas_ta, con nombres renombrados)
    - Bollinger Bands
    - Retrocesos de Fibonacci
    - Delta de Volumen (compra/venta aproximada)
    - Varias SMA en base a la temporalidad
    - Volatilidad anualizada
    - Manejo flexible de NaNs: solo se hace dropna de columnas críticas
    - Verificación de duplicados
    - Manejo de errores retornando None
    
    Args:
        df (pd.DataFrame): DataFrame con columnas ['timestamp','open','high','low','close','volume', ...]
        timeframe (str): Temporalidad ('1d','4h','1h','15m')
    
    Returns:
        pd.DataFrame o None en caso de error.
    """
    log_event("INFO", f"Iniciando cálculo de indicadores para {timeframe}", module="indicators")

    try:
        # ----------------------------------------------------------------------
        # 1. Configuración dinámica por temporalidad
        # ----------------------------------------------------------------------
        params = {
            '1d':  {'sma': [50, 200], 'fib_window': 90, 'volatility_window': 20},
            '4h':  {'sma': [20, 50],  'fib_window': 45, 'volatility_window': 14},
            '1h':  {'sma': [20, 50],  'fib_window': 30, 'volatility_window': 10},
            '15m': {'sma': [10, 20],  'fib_window': 15, 'volatility_window': 7},
        }
        cfg = params.get(timeframe, params['1h'])
        sma_periods = cfg['sma']
        
        # ----------------------------------------------------------------------
        # 2. Cálculos vectorizados básicos (returns, body, rango HL)
        # ----------------------------------------------------------------------
        df['returns'] = df['close'].pct_change()
        df['body'] = df['close'] - df['open']
        df['hl_range'] = df['high'] - df['low']

        # ----------------------------------------------------------------------
        # 3. RSI
        # ----------------------------------------------------------------------
        df['rsi'] = ta.rsi(df['close'], length=14)

        # ----------------------------------------------------------------------
        # 4. MACD (renombrando columnas)
        # ----------------------------------------------------------------------
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        # renombrar macd columns para evitar nombres como MACD_12_26_9, MACDs_12_26_9, etc.
        macd.columns = ['MACD_line', 'MACD_hist', 'MACD_signal']
        df = pd.concat([df, macd], axis=1)

        # ----------------------------------------------------------------------
        # 5. ATR y ADX (renombrando)
        # ----------------------------------------------------------------------
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        # Renombrar adx columns: 'ADX_14','DMP_14','DMN_14'
        adx_df.columns = ['ADX_value', 'DI_plus', 'DI_minus']
        
        df = pd.concat([df, atr_series.rename("ATR"), adx_df], axis=1)

        # ----------------------------------------------------------------------
        # 6. Bollinger Bands
        # ----------------------------------------------------------------------
        df['sma_20'] = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + 2 * rolling_std
        df['bb_lower'] = df['sma_20'] - 2 * rolling_std
        df['bb_ratio'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ----------------------------------------------------------------------
        # 7. Retrocesos de Fibonacci (rolling)
        # ----------------------------------------------------------------------
        max_price = df['high'].rolling(cfg['fib_window']).max()
        min_price = df['low'].rolling(cfg['fib_window']).min()
        total_range = max_price - min_price

        fib_levels = {
            'fib_0.236': max_price - total_range * 0.236,
            'fib_0.382': max_price - total_range * 0.382,
            'fib_0.5':   max_price - total_range * 0.5,
            'fib_0.618': max_price - total_range * 0.618,
            'fib_0.786': max_price - total_range * 0.786
        }

        for level, value in fib_levels.items():
            df[level] = value
            df[f'dist_to_{level}'] = (df['close'] - value) / (value.replace(0, np.nan)) * 100

        # ----------------------------------------------------------------------
        # 8. Delta de Volumen (buy vs sell aproximado)
        # ----------------------------------------------------------------------
        df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
        df['sell_volume'] = df['volume'] * (df['close'] <= df['open']).astype(int)
        df['volume_delta'] = df['buy_volume'] - df['sell_volume']
        df['volume_ratio'] = np.where(df['sell_volume'] == 0, np.inf, df['buy_volume'] / df['sell_volume'])

        # ----------------------------------------------------------------------
        # 9. SMA dinámicas
        # ----------------------------------------------------------------------
        for period in sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()

        # ----------------------------------------------------------------------
        # 10. Volatilidad anualizada (escala por timeframe)
        # ----------------------------------------------------------------------
        scaling_factors = {'1d': 252, '4h': 63, '1h': 24, '15m': 96}
        scale = scaling_factors.get(timeframe, 252)
        vol_window = cfg['volatility_window']
        df['volatility'] = df['returns'].rolling(vol_window).std() * np.sqrt(scale)

        # ----------------------------------------------------------------------
        # 11. Revisar duplicados en columnas
        # ----------------------------------------------------------------------
        if df.columns.duplicated().any():
            log_event("WARNING", "Columnas duplicadas detectadas en DataFrame", module="indicators")
        
        # ----------------------------------------------------------------------
        # 12. Manejo de NaNs
        #
        # Eliminamos filas que tengan NaN en RSI, MACD_line y MACD_signal
        # (consideramos que son columnas clave). Ajustar según tu criterio.
        # ----------------------------------------------------------------------
        df.dropna(subset=['rsi','MACD_line','MACD_signal'], inplace=True)

        # ----------------------------------------------------------------------
        # 13. Limpieza final (opcional)
        # ----------------------------------------------------------------------
        # Eliminamos columnas temporales si no las necesitamos
        df.drop(columns=['body','hl_range'], inplace=True, errors='ignore')

        log_event("INFO", f"Cálculo completado para {timeframe}. Features totales: {len(df.columns)}", module="indicators")

    except Exception as e:
        log_event("ERROR", f"Error en {timeframe}: {str(e)}", module="indicators")
        return None  # Manejo de error: retorna None para que el pipeline decida

    return df
