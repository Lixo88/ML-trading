import pandas as pd
import numpy as np
from datetime import timedelta
from prepare_data.logs import log_event, log_feature_stats

def load_timeframe_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Carga datos de un timeframe específico con validación estricta."""
    file_path = f"data_prepared/{timeframe}/{symbol}_{timeframe}_with_context.parquet"
    try:
        df = pd.read_parquet(file_path)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'rsi', 'macd', 'ATR', 'zone_relevance']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    except Exception as e:
        log_event("ERROR", f"Error cargando {timeframe}: {str(e)}", "feature_engineering")
        return pd.DataFrame()

def enrich_with_higher_tf(df_4h: pd.DataFrame, df_higher: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Enriquece datos 4H con información de timeframe superior usando merge temporal."""
    try:
        df_higher = df_higher.resample('4H').last().ffill()  # Alinear a velas 4H
        merged = pd.merge_asof(
            df_4h.sort_index(),
            df_higher.add_prefix(f'{tf}_'),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        return merged
    except Exception as e:
        log_event("ERROR", f"Error merge con {tf}: {str(e)}", "feature_engineering")
        return df_4h

def aggregate_lower_tf(df_4h: pd.DataFrame, df_lower: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Agrega datos de timeframe inferior a 4H con ventana móvil."""
    try:
        # Calcular ventanas de 4 horas (ej: 16 velas de 15m)
        window_size = {'1h': 4, '15m': 16}.get(tf, 1)
        
        lower_resampled = df_lower.resample('4H').agg({
            'volume': ['sum', 'max'],
            'ATR': 'mean',
            'zone_relevance': lambda x: x.max(skipna=True)
        })
        lower_resampled.columns = [f'{tf}_vol_sum', f'{tf}_vol_max', 
                                  f'{tf}_ATR_mean', f'{tf}_zone_relevance_max']
        
        # Merge con datos 4H
        return df_4h.merge(lower_resampled, left_index=True, right_index=True, how='left')
    except Exception as e:
        log_event("ERROR", f"Error agregando {tf}: {str(e)}", "feature_engineering")
        return df_4h

def calculate_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering específico para 4H."""
    try:
        # 1. Momentum y Volatilidad
        df['returns_4h'] = df['close'].pct_change().shift(1)
        df['volatility_4h'] = df['returns_4h'].rolling(20).std() * np.sqrt(252/6)  # Anualizada
        
        # 2. Interacción entre features
        df['liquidity_ratio'] = df['volume'] / df['15m_vol_sum'].replace(0, 1e-6)
        df['zone_impact'] = df['zone_relevance'] * df['1d_ATR_mean']
        
        # 3. Diferencias porcentuales con 1D
        df['sma_200_deviation'] = (df['close'] - df['1d_sma_200']) / df['1d_sma_200']
        
        # 4. Señales compuestas
        df['macro_micro_signal'] = np.where(
            (df['1d_market_phase'] == 1) & (df['rsi'] < 40),
            df['1h_zone_relevance_max'] * 0.7 + df['15m_vol_max'] * 0.3,
            np.nan
        )
        
        return df.dropna()
    except Exception as e:
        log_event("ERROR", f"Error en feature engineering: {str(e)}", "feature_engineering")
        return pd.DataFrame()

def process_symbol(symbol: str):
    """Pipeline completo para un símbolo."""
    log_event("INFO", f"Iniciando procesamiento de {symbol}", "feature_engineering")
    
    try:
        # 1. Cargar datos base 4H
        df_4h = load_timeframe_data(symbol, '4h')
        if df_4h.empty:
            return

        # 2. Enriquecer con 1D
        df_1d = load_timeframe_data(symbol, '1d')
        df_4h = enrich_with_higher_tf(df_4h, df_1d, '1d')

        # 3. Agregar datos de 1H y 15m
        for tf in ['1h', '15m']:
            df_lower = load_timeframe_data(symbol, tf)
            df_4h = aggregate_lower_tf(df_4h, df_lower, tf)

        # 4. Feature Engineering
        df_final = calculate_4h_features(df_4h)
        
        # 5. Validación final
        log_feature_stats(df_final, [
            'returns_4h', 'volatility_4h', 'liquidity_ratio',
            'zone_impact', 'macro_micro_signal'
        ], "feature_engineering")
        
        # 6. Guardar dataset
        output_path = f"data_ml/{symbol}/4h_dataset_v2.parquet"
        df_final.to_parquet(output_path)
        log_event("SUCCESS", f"Dataset guardado en {output_path}", "feature_engineering")

    except Exception as e:
        log_event("ERROR", f"Error procesando {symbol}: {str(e)}", "feature_engineering")

def main():
    """Ejecución paralela para múltiples símbolos."""
    from joblib import Parallel, delayed
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Configurable
    Parallel(n_jobs=4, verbose=10)(
        delayed(process_symbol)(symbol) for symbol in symbols
    )

if __name__ == "__main__":
    main()