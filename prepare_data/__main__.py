import os
import logging
from download_data.validation import validate_and_correct_data
from prepare_data.indicators import calculate_technical_indicators
from prepare_data.zones import identify_supply_demand_zones
from prepare_data.context import validate_cross_temporal_zones, integrate_key_events, calculate_zone_relevance
from prepare_data.labeling import label_events, add_event_metadata, label_zone_strength, label_zone_type, finalize_labels
from prepare_data.utils import save_processed_data
from prepare_data.logs import log_event, configure_logging

# Configuración inicial
data_dir = "data"
output_dir = "data_processed"
configure_logging()

log_event("INFO", "Inicio del proceso de preparación de datos", module="main")

def process_symbol(symbol_dir):
    symbol_path = os.path.join(data_dir, symbol_dir)
    if not os.path.isdir(symbol_path):
        return

    for timeframe_file in os.listdir(symbol_path):
        input_filepath = os.path.join(symbol_path, timeframe_file)
        output_filepath = os.path.join(output_dir, symbol_dir, timeframe_file.replace(".parquet", "_processed.parquet"))

        log_event("INFO", f"Procesando archivo: {input_filepath}", module="main")

        try:
            # Paso 1: Validar y corregir datos
            df = validate_and_correct_data(input_filepath)

            # Determinar timeframe
            timeframe = "1d" if "1d" in timeframe_file else "4h" if "4h" in timeframe_file else "1h" if "1h" in timeframe_file else "15m"
            
            # Paso 2: Calcular indicadores técnicos
            df = calculate_technical_indicators(df, timeframe)
            
            # Paso 3: Identificar zonas de oferta y demanda
            if timeframe in ["4h", "1d"]:
                df = identify_supply_demand_zones(df)
                
            # Paso 4: Validar zonas entre temporalidades (4H y 1D)
            if timeframe == "4h":
                df_1d_filepath = os.path.join(output_dir, symbol_dir, timeframe_file.replace("4h", "1d"))
                if os.path.exists(df_1d_filepath):
                    df_1d = validate_and_correct_data(df_1d_filepath)
                    df, df_1d = validate_cross_temporal_zones(df, df_1d)
                    save_processed_data(df_1d, df_1d_filepath)
                    log_event("INFO", "Validación cruzada 4H-1D completada", module="main")
            
            # Paso 5: Integrar eventos clave y calcular relevancia de zonas
            df = integrate_key_events(df, [])  # Se debe definir la lista de eventos clave
            df = calculate_zone_relevance(df)
            
            # Paso 6: Etiquetado de eventos y zonas
            df = label_events(df)
            df = add_event_metadata(df)
            df = label_zone_strength(df)
            df = label_zone_type(df)
            df = finalize_labels(df)
            
            # Paso 7: Guardar datos procesados
            save_processed_data(df, output_filepath)
            log_event("INFO", f"Datos procesados guardados en: {output_filepath}", module="main")
        
        except Exception as e:
            log_event("ERROR", f"Error procesando {input_filepath}: {str(e)}", module="main")

log_event("INFO", "Proceso de preparación de datos completado", module="main")

if __name__ == "__main__":
    for symbol_dir in os.listdir(data_dir):
        process_symbol(symbol_dir)
