# prepare_data/__main__.py

import os
import sys
import argparse
import glob
import re
import pandas as pd

# Imports de tus módulos locales
from prepare_data.indicators import calculate_technical_indicators
from prepare_data.zones import run_zones_pipeline
from prepare_data.logs import log_event

def main():
    """
    Orquestador principal para la etapa 'prepare_data'.
    - 1) Encontrar archivos '_validated.parquet'
    - 2) Calcular indicadores -> guarda '_with_indicators.parquet'
    - 3) Detectar zonas -> guarda '_with_zones.parquet' y '_zones_clusters.parquet'
    - 4) (a futuro) context, labeling...
    """
    parser = argparse.ArgumentParser(description="Orquestador prepare_data")
    parser.add_argument("--symbol", type=str, help="Símbolo, ej: BTCUSDT", required=False)
    parser.add_argument("--timeframe", type=str, help="Timeframe, ej: 4h, 1d", required=False)
    parser.add_argument("--data_dir", type=str, default="data_processed", help="Directorio base donde están los datos")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    data_dir = args.data_dir

    # 1) Buscar archivos validado
    # Ejemplo: <symbol>_<timeframe>_..._validated.parquet
    pattern = "*_validated.parquet"
    if symbol and timeframe:
        # filtrar también por symbol y timeframe
        pattern = f"{symbol}_{timeframe}_*_validated.parquet"

    validated_files = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    if not validated_files:
        print(f"No se encontraron archivos que coincidan con: {pattern} en {data_dir}")
        sys.exit(0)

    for filepath in validated_files:
        # Ej: BTCUSDT_4h_20210101_to_20250201_validated.parquet
        basename = os.path.basename(filepath)
        # Reemplazar _validated por _with_indicators, etc.
        # (Podemos usar una regex, o un string replace)
        file_no_ext = basename.replace(".parquet","")
        base_name_ind = file_no_ext.replace("_validated","_with_indicators")
        base_name_zon = file_no_ext.replace("_validated","_with_zones")
        base_name_zon_clust = file_no_ext.replace("_validated","_zones_clusters")

        # Determinar carpeta de salida (similar a la carpeta donde está el validated)
        dir_path = os.path.dirname(filepath)

        output_indicators_path = os.path.join(dir_path, f"{base_name_ind}.parquet")
        output_zones_path      = os.path.join(dir_path, f"{base_name_zon}.parquet")
        output_clusters_path   = os.path.join(dir_path, f"{base_name_zon_clust}.parquet")

        log_event("INFO", f"Procesando: {filepath}", module="main_prepare_data")

        # 2) Leer df validado
        df_valid = pd.read_parquet(filepath)

        # (Opción) Inferir timeframe si no se pasó
        # pero si ya se hace con symbol/timeframe, se omite

        # 3) Calcular INDICATORS
        log_event("INFO", "Calculando indicadores...", module="main_prepare_data")
        df_ind = calculate_technical_indicators(df_valid, timeframe or infer_timeframe_from_name(basename))
        if df_ind is None:
            log_event("ERROR", "Fallo en calculate_technical_indicators, skip archivo", module="main_prepare_data")
            continue

        # Guardar _with_indicators
        df_ind.to_parquet(output_indicators_path, index=False)
        log_event("INFO", f"Guardado: {output_indicators_path}", module="main_prepare_data")

        # 4) Detectar ZONAS
        log_event("INFO", "Detectando zonas...", module="main_prepare_data")
        df_main, df_zones = run_zones_pipeline(df_ind)

        # Guardar df_main => _with_zones
        df_main.to_parquet(output_zones_path, index=False)
        log_event("INFO", f"Guardado: {output_zones_path}", module="main_prepare_data")

        # Guardar df_zones => _zones_clusters
        df_zones.to_parquet(output_clusters_path, index=False)
        log_event("INFO", f"Guardado: {output_clusters_path}", module="main_prepare_data")

        # Podrías continuar con context, labeling, etc. en otros pasos
        log_event("INFO", f"Pipeline completado para {basename}", module="main_prepare_data")


def infer_timeframe_from_name(filename: str) -> str:
    """
    Pequeño helper para extraer la TF (1d,4h,1h,15m) del nombre del archivo
    si no se pasa como argumento.
    """
    fname = filename.lower()
    if "1d" in fname:
        return "1d"
    elif "4h" in fname:
        return "4h"
    elif "1h" in fname:
        return "1h"
    elif "15m" in fname:
        return "15m"
    return "1h"  # fallback

if __name__ == "__main__":
    main()


#HAY QUE CONTINUARLO PARA INTEGRAR CONTEXT, LABELING Y TMB LOG Y UTILS