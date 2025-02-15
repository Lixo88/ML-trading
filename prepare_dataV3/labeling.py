"""
labeling.py

Genera un único dataset final para entrenar un modelo (XGBoost u otro).
1) Carga el timeframe principal (p.ej. 4H) con _with_context.parquet + merges TFs (1D,1H,15m).
2) Feature engineering adicional (returns_4h, macro_micro_signal, zone_impact, etc.).
3) Etiquetado de eventos (rebote, rompimiento, falso breakout) según config.
4) Guarda symbol_startdate-enddate_prepared.parquet => input directo para ML.

Los parámetros cruciales (rebote_window, breakout_threshold, etc.)
van en config['labeling'] y pueden ser refinados por un submodelo 
o backtesting en el futuro.
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from joblib import Parallel, delayed
from prepare_data.logs import log_event, log_feature_stats

def load_timeframe_data(symbol: str, timeframe: str, config: dict) -> pd.DataFrame:
    """
    Carga los datos _with_context de la TF dada y retorna un DF indexado por timestamp.
    Espera un path: data_prepared/{timeframe}/SYMBOL_{timeframe}_with_context.parquet
    (Ajustar segun tu carpeta real).
    """
    module = "labeling"
    base_dir = config.get('data_prepared_dir', "data_prepared")
    file_name = f"{symbol}_{timeframe}_with_context.parquet"
    file_path = os.path.join(base_dir, timeframe, file_name)

    try:
        df = pd.read_parquet(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        log_event("INFO", f"Cargado {file_path} => filas={len(df)}", module)
        return df
    except Exception as e:
        log_event("ERROR", f"No se pudo cargar {file_path}: {str(e)}", module)
        return pd.DataFrame()

def enrich_with_higher_tf(df_main: pd.DataFrame, df_higher: pd.DataFrame, prefix: str, config: dict) -> pd.DataFrame:
    """
    Enriquecer df_main con columns del TF superior via merge_asof (backward).
    E.g. prefix='1d_'
    """
    module = "labeling"
    if df_higher.empty or df_main.empty:
        return df_main

    try:
        # Resample o no, segun la convención. 
        # Asumimos df_higher se freq mayor (1D) => rolling a la freq del main (4H).
        freq_main = config.get('labeling', {}).get('main_timeframe_freq', '4H')
        df_higher_res = df_higher.resample(freq_main).last().ffill()

        merged = pd.merge_asof(
            df_main.sort_index(),
            df_higher_res.sort_index().add_prefix(prefix),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        log_event("INFO", f"Enriquecido con TF superior {prefix}, filas={len(merged)}", module)
        return merged
    except Exception as e:
        log_event("ERROR", f"Error en enrich_with_higher_tf: {str(e)}", module)
        return df_main

def aggregate_lower_tf(df_main: pd.DataFrame, df_lower: pd.DataFrame, prefix: str, config: dict) -> pd.DataFrame:
    """
    Agrega TF inferior al main TF (e.g. 1H, 15m -> 4H) con agg. 
    E.g. sum de volume, max zone_relevance, etc.
    """
    module = "labeling"
    if df_lower.empty or df_main.empty:
        return df_main

    try:
        freq_main = config.get('labeling', {}).get('main_timeframe_freq', '4H')
        # Por default, rolling aggregator
        # Ej. si prefix='1h', 4H = 4 velas de 1h => sum, max, etc.
        # Ajustar segun necesites
        aggregator = {
            'volume': ['sum','max'],
            'zone_relevance': 'max'
        }
        # Podrias meter RSI, etc. "mean"

        lower_resampled = df_lower.resample(freq_main).agg(aggregator)
        # renombrar
        lower_resampled.columns = [f"{prefix}_{col[0]}_{col[1]}" for col in lower_resampled.columns]

        merged = pd.merge_asof(
            df_main.sort_index(),
            lower_resampled.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        log_event("INFO", f"Aggregate TF inferior {prefix}, filas={len(merged)}", module)
        return merged
    except Exception as e:
        log_event("ERROR", f"Error en aggregate_lower_tf: {str(e)}", module)
        return df_main

def feature_engineering_main(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Crea features finales: returns, volatility, combos, etc.
    """
    module = "labeling"
    if df.empty:
        return df

    try:
        # 1) Returns
        df['returns_main'] = df['close'].pct_change().shift(1)
        # 2) Volatilidad
        df['volatility_main'] = df['returns_main'].rolling(20).std() * np.sqrt(252*6)  # suposic

        # 3) Ejemplo synergy
        # Ej: zone_impact = zone_relevance * 1d_ATR?
        if '1d_ATR' in df.columns and 'zone_relevance' in df.columns:
            df['zone_impact'] = df['zone_relevance'] * df['1d_ATR']

        # 4) Log stats
        log_feature_stats(df, ['returns_main','volatility_main','zone_impact'], module)
        return df
    except Exception as e:
        log_event("ERROR", f"feature_engineering_main error: {str(e)}", module)
        return df

def label_events(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Asigna label (0=none,1=rebote,2=breakout,3=falso_break) segun la logica:
     - rebote_window, rebote_threshold => si en N velas sube +X% sin romper
     - breakout => ...
     - etc.
    Placeholder. Ajustar la logica concreta.
    """
    module = "labeling"
    if df.empty:
        df['label'] = 0
        return df

    label_cfg = config.get('labeling', {})
    rebote_window = label_cfg.get('rebote_window', 3)
    rebote_thresh = label_cfg.get('rebote_threshold', 0.015)  # 1.5%
    breakout_window = label_cfg.get('breakout_window', 3)
    breakout_thresh = label_cfg.get('breakout_threshold', 0.01)
    # etc

    df['label'] = 0  # default none

    # Ejemplo: detectar REBOTE en DEMANDA
    # Pseudocódigo:
    # for i in range(len(df)):
    #   if df.iloc[i]['closest_demand_relevance']>0.2:
    #       zone_lower = ...
    #       price_in = df.iloc[i]['close']
    #       # examino proximas rebote_window velas => sube?
    #       # if sube + rebote_thresh y no cierra < zone_lower => label=1
    # ...

    # Minimally implement placeholder:
    for i in range(len(df)-rebote_window):
        if df.iloc[i]['closest_demand_relevance']>0.2:
            z_low = df.iloc[i]['low']  # or actual zone lower
            entry_price = df.iloc[i]['close']
            # check next N => max_close
            look_ahead = df.iloc[i+1:i+rebote_window+1]
            max_close = look_ahead['close'].max()
            # if max_close >= entry_price*(1+rebote_thresh):
            #   df.iloc[i, df.columns.get_loc('label')] = 1

    log_event("INFO", f"Etiquetado final => rebote/breakout => ver docstring", module)
    return df

def process_symbol(symbol: str, config: dict):
    """
    Pipeline final para 1 símbolo. 
     1) Carga main timeframe (4H) y merges con 1D, 1H, 15m
     2) Feature engineering
     3) Etiquetado
     4) Guarda symbol_startdate-enddate_prepared.parquet
    """
    module = "labeling"
    log_event("INFO", f"Procesando {symbol}", module)

    labeling_cfg = config.get('labeling', {})
    main_tf = labeling_cfg.get('main_timeframe', '4h')
    # supongamos date strings
    start_date = labeling_cfg.get('start_date','2021-01-01')
    end_date   = labeling_cfg.get('end_date','2022-01-01')

    # 1) Cargar main TF
    df_main = load_timeframe_data(symbol, main_tf, config)
    if df_main.empty:
        return

    # 2) Enriquecer con 1D
    tf_higher = labeling_cfg.get('higher_tf', '1d')
    df_higher = load_timeframe_data(symbol, tf_higher, config)
    df_main = enrich_with_higher_tf(df_main, df_higher, f"{tf_higher}_", config)

    # 3) Agregar lower TF 1h, 15m
    lower_tfs = labeling_cfg.get('lower_tfs', ['1h','15m'])
    for ltf in lower_tfs:
        df_low = load_timeframe_data(symbol, ltf, config)
        df_main = aggregate_lower_tf(df_main, df_low, ltf, config)

    # 4) Feature engineering final
    df_main = feature_engineering_main(df_main, config)
    if df_main.empty:
        return

    # 5) Etiquetado
    df_main = label_events(df_main, config)

    # 6) Guardar => symbol_startdate-enddate_prepared.parquet
    out_dir = labeling_cfg.get('output_dir', "data_ml")
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{symbol}_{start_date}_to_{end_date}_prepared.parquet"
    out_path = os.path.join(out_dir, out_file)
    df_main.reset_index(inplace=True)
    df_main.to_parquet(out_path, index=False)
    log_event("INFO", f"Dataset final guardado => {out_path}", module)

def main():
    """Ejecución en paralelo para varios símbolos."""
    from joblib import Parallel, delayed

    # Podrías cargar config de un .yaml
    import yaml
    with open("prepare_data_config.yaml","r") as f:
        config = yaml.safe_load(f)

    symbols = config.get('symbols', ['BTCUSDT'])
    # Paralelo 
    Parallel(n_jobs=2, verbose=5)(
        delayed(process_symbol)(symbol, config) for symbol in symbols
    )

if __name__ == "__main__":
    main()
