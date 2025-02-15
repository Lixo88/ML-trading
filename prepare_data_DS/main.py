# prepare_data/main.py
import os
import sys
import argparse
import yaml
import pandas as pd
from typing import Dict, List
from joblib import Parallel, delayed
from .logs import logger
from .indicators import calculate_technical_indicators
from .zones import run_zones_pipeline
from .context import run_context_pipeline

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.log_event('INFO', f"Config loaded from {config_path}", "main")
        return config
    except Exception as e:
        logger.log_event('ERROR', f"Config error: {str(e)}", "main")
        raise

def process_timeframe(symbol: str, tf: str, config: Dict, data_dir: str, output_dir: str):
    try:
        # 1. Cargar datos validados
        validated_path = os.path.join(data_dir, symbol, f"{symbol}_{tf}_validated.parquet")
        df = pd.read_parquet(validated_path)
        
        # 2. Calcular indicadores
        df_indicators = calculate_technical_indicators(df, tf, config['indicators'])
        if df_indicators is None:
            raise ValueError("Technical indicators failed")
        
        # 3. Detección de zonas
        df_main_zones, df_zones = run_zones_pipeline(df_indicators, config['zones'])
        
        # 4. Contexto multi-TF
        higher_tf = get_higher_timeframe(tf)
        df_zones_higher = load_higher_tf_data(symbol, higher_tf, output_dir)
        df_final = run_context_pipeline(df_main_zones, df_zones, df_zones_higher, config['context'])
        
        # 5. Guardar resultados
        save_path = os.path.join(output_dir, symbol, tf)
        os.makedirs(save_path, exist_ok=True)
        
        df_final.to_parquet(os.path.join(save_path, f"{symbol}_{tf}_processed.parquet"))
        logger.log_event('INFO', f"Saved {tf} data for {symbol}", "main",
                        {"path": save_path, "rows": len(df_final)})
        
        return True
    except Exception as e:
        logger.log_event('ERROR', f"Failed processing {tf} for {symbol}: {str(e)}", "main")
        return False

def get_higher_timeframe(tf: str) -> str:
    hierarchy = ['15m', '1h', '4h', '1d']
    try:
        return hierarchy[hierarchy.index(tf) + 1]
    except IndexError:
        return None

def load_higher_tf_data(symbol: str, tf: str, output_dir: str) -> pd.DataFrame:
    if not tf:
        return None
    path = os.path.join(output_dir, symbol, tf, f"{symbol}_{tf}_processed.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None

def process_symbol(symbol: str, config: Dict, data_dir: str, output_dir: str):
    logger.log_event('INFO', f"Starting processing for {symbol}", "main")
    timeframes = config['timeframes']
    
    results = Parallel(n_jobs=4)(delayed(process_timeframe)(
        symbol, tf, config, data_dir, output_dir
    ) for tf in timeframes)
    
    if all(results):
        logger.log_event('INFO', f"Completed {symbol} successfully", "main")
    else:
        logger.log_event('WARNING', f"Completed {symbol} with errors", "main",
                        {"success_rate": f"{sum(results)/len(results):.0%}"})

def main():
    parser = argparse.ArgumentParser(description="Trading Data Pipeline")
    parser.add_argument(
        '--config', 
        default='prepare_data_config.yaml',
        help='Ruta al archivo de configuración principal (default: prepare_data_config.yaml)'
    )
    parser.add_argument(
        '--data_dir', 
        default='data/validated',
        help='Directorio de datos validados (default: data/validated)'
    )
    parser.add_argument(
        '--output_dir', 
        default='data/processed',
        help='Directorio de salida procesado (default: data/processed)'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['BTCUSDT'],
        help='Símbolos a procesar separados por espacios (default: BTCUSDT)'
    )
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        os.makedirs(args.output_dir, exist_ok=True)
        
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Directorio de datos no encontrado: {args.data_dir}")
        
        Parallel(n_jobs=2)(delayed(process_symbol)(
            symbol, config, args.data_dir, args.output_dir
        ) for symbol in args.symbols)
        
        logger.log_event("INFO", "Pipeline ejecutado exitosamente", "main")
        
    except Exception as e:
        logger.log_event("CRITICAL", f"Error fatal: {str(e)}", "main")
        sys.exit(1)

if __name__ == "__main__":
    main()