import os
import sys
import glob
import argparse
import yaml
import pandas as pd

from prepare_data.logs import log_event
from prepare_data.indicators import calculate_technical_indicators
from prepare_data.zones import run_zones_pipeline
from prepare_data.context import run_context_pipeline

def load_config(config_path: str):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No se encontró config en {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Orquestador multi-TF (4H validado con 1D).")
    parser.add_argument("--config", default="prepare_data_config.yaml", help="Archivo YAML de config.")
    parser.add_argument("--data_dir", default="data_processed", help="Carpeta base de data validada.")
    parser.add_argument("--output_dir", default="data_prepared", help="Carpeta para guardar outputs.")
    parser.add_argument("--symbol", type=str, required=True, help="Símbolo (ej: BTCUSDT).")
    args = parser.parse_args()

    config = load_config(args.config)
    log_event("INFO", f"Cargada config desde {args.config}", module="main")

    data_dir = args.data_dir
    out_dir  = args.output_dir
    sym      = args.symbol

    # Creamos subcarpetas
    ind_out_dir   = os.path.join(out_dir, "indicators")
    zones_out_dir = os.path.join(out_dir, "zones")
    ctx_out_dir   = os.path.join(out_dir, "context")

    for d in [ind_out_dir, zones_out_dir, ctx_out_dir]:
        os.makedirs(d, exist_ok=True)

    # Queremos procesar 4H y 1D y luego validar 4H con 1D en context
    timeframes_to_process = ["4h", "1d"]

    # Sub-config
    indicators_cfg = config.get("indicators", {})
    zones_cfg      = config.get("zones", {})
    context_cfg    = config.get("context", {})

    # Diccionario donde guardaremos los df_main / df_zones de c/timeframe
    results_zones = {}  # p.ej. results_zones["4h"] = (df_main_zones_4h, df_zones_4h)
    results_ind   = {}  # p.ej. results_ind["4h"] = df_ind_4h

    for tf in timeframes_to_process:
        # 1) Buscar _validated.parquet
        pattern = f"{sym}_{tf}_*_validated.parquet"
        search_path = os.path.join(data_dir, sym, pattern)
        files = glob.glob(search_path)

        if not files:
            log_event("WARNING", f"No hay '{pattern}' en {os.path.join(data_dir, sym)}", module="main")
            continue

        # Tomamos el primero si hay varios. (Podrías manejar loops si deseas)
        validated_file = files[0]
        filename = os.path.basename(validated_file)
        log_event("INFO", f"Procesando => {filename}", module="main")

        df_valid = pd.read_parquet(validated_file)
        if df_valid.empty:
            log_event("WARNING", f"{filename} está vacío, saltando...", module="main")
            continue

        # 2) INDICATORS
        df_ind = calculate_technical_indicators(df_valid, tf, indicators_cfg)
        if df_ind is None:
            log_event("ERROR", f"Fallo en indicadores para {sym} {tf}.", module="main")
            continue

        out_ind_name = filename.replace("_validated.parquet","_with_indicators.parquet")
        out_ind_path = os.path.join(ind_out_dir, out_ind_name)
        df_ind.to_parquet(out_ind_path, index=False)
        log_event("INFO", f"Guardado => {out_ind_path}", module="main")

        results_ind[tf] = df_ind

        # 3) ZONES
        df_main_zones, df_zones_clusters = run_zones_pipeline(df_ind, zones_cfg)

        out_z_main_name = out_ind_name.replace("_with_indicators.parquet","_with_zones.parquet")
        out_z_clust_name= out_ind_name.replace("_with_indicators.parquet","_zones_clusters.parquet")

        out_z_main_path = os.path.join(zones_out_dir, out_z_main_name)
        out_z_clust_path= os.path.join(zones_out_dir, out_z_clust_name)

        df_main_zones.to_parquet(out_z_main_path, index=False)
        df_zones_clusters.to_parquet(out_z_clust_path, index=False)
        log_event("INFO", f"Guardado => {out_z_main_path} y {out_z_clust_path}", module="main")

        results_zones[tf] = (df_main_zones, df_zones_clusters)

    # Ahora tenemos results_zones["4h"] y results_zones["1d"] (si existieron)
    if "4h" not in results_zones or "1d" not in results_zones:
        log_event("WARNING", "No se pudo procesar ambos TF (4h y 1d). Falta uno. No se hará validación multiTF", module="main")
        sys.exit(0)

    # Extraemos
    df_main_4h, df_zones_4h = results_zones["4h"]
    df_main_1d, df_zones_1d = results_zones["1d"]

    # 4) CONTEXT => validación multiTF => 4H con 1D
    log_event("INFO", "Corriendo context: validando 4H con 1D, integrando relevancia...", module="main")

    # Llamamos context con df_main_4h, df_zones_4h como "df_zones_lower",
    # y df_zones_higher = df_zones_1d
    # key_events=None (por ahora)

    # Ajustar config para forzar multi_tf_validation=true
    context_cfg["multi_tf_validation"] = True

    df_main_ctx_4h, df_zones_ctx_4h = run_context_pipeline(
        df_main_4h,
        df_zones_4h,
        df_zones_higher=df_zones_1d,
        key_events=None,
        config=context_cfg
    )

    # GUARDAR => 4H con context validado
    # Nombre base? Tomamos algo de out_z_main_name
    # p.ej. "BTCUSDT_4h_20200101_to_20210101_with_context.parquet"

    out_ctx_main_4h = out_z_main_name.replace("_with_zones.parquet","_with_context.parquet")
    out_ctx_zones_4h= out_z_clust_name.replace("_zones_clusters.parquet","_zones_context.parquet")

    out_ctx_main_4h_path = os.path.join(ctx_out_dir, out_ctx_main_4h)
    out_ctx_zones_4h_path= os.path.join(ctx_out_dir, out_ctx_zones_4h)

    df_main_ctx_4h.to_parquet(out_ctx_main_4h_path, index=False)
    df_zones_ctx_4h.to_parquet(out_ctx_zones_4h_path, index=False)

    log_event("INFO", f"Guardado => {out_ctx_main_4h_path} y {out_ctx_zones_4h_path}", module="main")

    # Opcional: correr context en 1D en modo "single TF"
    # (Si lo deseas: df_main_ctx_1d, df_zones_ctx_1d = run_context_pipeline(...))
    # ...
    
    log_event("INFO", "Pipeline completado con validación 4H vs 1D!", module="main")


if __name__ == "__main__":
    main()
