# trading_bot_ml
Para que aprenda la maquinola aprenda
DIOS QUE DOLOR DE CABEZA

A. download_data PACKAGE
Este proyecto es la primera fase de un sistema de Machine Learning aplicado al trading. Su objetivo es descargar, validar y preparar datos históricos de mercados financieros desde la API de Binance, para su posterior uso en la construcción y entrenamiento de modelos predictivos.

Características Principales
Descarga de datos históricos: Obtiene datos OHLCV (Open, High, Low, Close, Volume) de múltiples símbolos y temporalidades.

Validación automática: Detecta y corrige gaps en los datos, valores faltantes y anomalías.

Preparación para ML: Los datos se almacenan en formato Parquet, optimizado para análisis y procesamiento en pipelines de Machine Learning.

Ejecución paralela: Descarga y procesa múltiples símbolos y temporalidades de manera concurrente.

Manejo robusto de errores: Reintentos automáticos y registro detallado de errores.

Configuración centralizada: Todos los parámetros se gestionan a través de un archivo YAML.

Dependencias: binance-client, pandas, pyarrow, pyyaml

Configuración
Edita el archivo config.yaml para especificar:

API Key y Secret: Credenciales de Binance API.

Símbolos: Lista de pares de trading (ej: BTCUSDT, ETHUSDT).

Temporalidades: Intervalos de tiempo (ej: 1d, 4h, 1h, 15m).

Rango de fechas: Fecha inicial y opcionalmente fecha final.

Directorios: Rutas para almacenar datos y logs.

Ejemplo de configuración:

yaml
Copy
api_key: "tu_api_key"
api_secret: "tu_api_secret"

symbols: #Se pueden agregar los que sean necesarios
  - BTCUSDT
  - ETHUSDT

timeframes: #mantener los 4 para 
  - "1d"
  - "4h"
  - "1h"
  - "15m"

start_date: "2021-01-01"
end_date: "2025-02-01"

output_dir: "data"
log_dir: "logs"
Uso
Instala las dependencias:

bash
Copy
pip install binance-client pandas pyarrow pyyaml
Configura el archivo config.yaml con tus parámetros.

Ejecuta el script principal:
Los datos descargados y validados estarán disponibles en la carpeta data/, organizados por símbolo.

Los datos se almacenan en formato columnar (Parquet) para eficiencia.

Se mantiene un registro detallado de todo el proceso.

Los datos están listos para la fase de feature engineering.

B. prepare_data/ PACKAGE
Objetivo Principal:
Este paquete implementa la etapa de preparación de datos para un sistema de trading algorítmico. Dado un archivo _validated.parquet (o varios) con datos OHLCV de un símbolo/timeframe, se encarga de:
Calcular indicadores técnicos (indicators.py).
Detectar y unificar zonas de oferta/demanda (zones.py).
Enriquecer la información con validaciones multi-timeframe y relevancias (context.py).
Asignar etiquetas de eventos en velas (rebotes, rupturas, etc.) mediante labeling.py.
Producir uno o varios DataFrames (o archivos .parquet) con toda la información lista para entrenar un modelo de ML (o para su uso en vivo).
1. Estructura General

prepare_data/
├─ __main__.py (o main.py)
├─ indicators.py
├─ zones.py
├─ context.py
├─ labeling.py (opcional/presente)
├─ logs.py
└─ prepare_data_config.ymal

El paquete prepare_data/ contiene los siguientes scripts y archivos principales:
logs.py
Manejo centralizado de logs.
Define la función log_event(level, msg, module) que escribe en prepare_data.log y opcionalmente con colores en consola.
No requiere inputs de config, se limita a formatear y registrar mensajes.
utils.py (Opcional/Extendible)
Funciones auxiliares, por ejemplo load_config() para cargar un YAML, cross_join genérico, etc.
Puede estar vacío o muy pequeño al inicio, crecerá a medida que se requieran más funciones repetitivas.
indicators.py
Cálculo de indicadores técnicos en un DataFrame con columnas [open, high, low, close, volume, timestamp].
Usa librerías como pandas_ta para RSI, MACD, ADX, ATR, Bollinger, retrocesos de Fibonacci, etc.
Devuelve un DataFrame con las nuevas columnas (rsi, MACD_line, fib_0.382, volume_delta, etc.) listo para las fases siguientes.
Logs con module="indicators".
Variables como SMA, ventana de fib, etc. se extraen desde un config (normalmente pasado por el orquestador).
zones.py
Detección de zonas de oferta y demanda, basadas en pivots, volumen, filtros de mecha/body, ATR, etc.
Unificación de pivots con DBSCAN (clustering) para producir zonas consolidadas.
Cálculo de zone_raw_strength (submodelo placeholder o pesos fijos) que integra volumen, repeticiones, frescura, etc.
Produce dos DataFrames:
df_main (mismo shape que el input, 1 fila por vela, con nuevas columnas demand_zone_lower, etc.).
df_zones (zona-based), con 1 fila por cluster unificado.
Logs con module="zones".
context.py
Validación multi-timeframe: (ej. 4H vs 1D) para marcar validated_by_higher = True cuando las zonas 4H se solapan con las de 1D.
Integración de key events (placeholder si use_key_events es False) para asignar event_impact.
Cálculo de zone_relevance, mezclando zone_raw_strength + event_impact + validación multiTF.
Asignación opcional de la zona más fuerte a cada vela (closest demand/supply).
Retorna:
df_main_context (vela-based) con columnas como closest_demand_relevance.
df_zones_context (zona-based) con zone_relevance.
Logs con module="context".
labeling.py (pendiente o ya integrado)
Etiqueta las velas con eventos concretos (rebote, ruptura, consolidación, etc.) en función de las zonas u otras señales.
Añade columnas como event_type o event_label.
Produce un DataFrame final _labeled.parquet con 1 fila por vela y la etiqueta que se usará en ML.
Logs con module="labeling".
__main__.py (o main.py)
Orquestador que:
Lee la config (prepare_data_config.yaml) y argumentos CLI.
Para cada símbolo/timeframe, busca *_validated.parquet, llama secuencialmente:
indicators.py → _with_indicators.parquet
zones.py → _with_zones.parquet, _zones_clusters.parquet
context.py → _with_context.parquet, _zones_context.parquet
(opcional) labeling.py → _labeled.parquet
Guarda salidas en subcarpetas (data_prepared/indicators, data_prepared/zones, etc.).
Lanza logs con module="main".

2. Flujo Lógico del Pipeline
El flujo típico, orchestrado por main.py, es:
Leer un parquet _validated.parquet (descargado + validado desde otra fase) con OHLCV.
Indicators:
calculate_technical_indicators(df, timeframe, config["indicators"]) => añade RSI, MACD, ATR, Bollinger, etc.
Guarda _with_indicators.parquet (en data_prepared/indicators/).
Zones:
run_zones_pipeline(df_ind, config["zones"]) => detecta pivots y unifica.
Devuelve (df_main_zones, df_zones_clusters), que se guardan como _with_zones.parquet y _zones_clusters.parquet (en data_prepared/zones/).
Context:
run_context_pipeline(df_main_zones, df_zones_clusters, df_zones_higher=?, key_events=?, config=...) =>
Valida multiTF si se pasa un “df_zones_higher” (p. ej. 1D).
Integra “event_impact” (placeholder) y calcula “zone_relevance”.
Asigna zona a cada vela, produce df_main_context y df_zones_context.
Se guardan en data_prepared/context/.
Labeling (futuro/presente):
Toma df_main_context y asigna la etiqueta final a cada vela (rebote, ruptura...).
Guarda _labeled.parquet.
El ML final usará _labeled.parquet como dataset.
3. Configuración por Submódulos
Cada uno de estos scripts puede requerir parámetros distintos (ventanas, umbrales, multiplicadores, etc.). Para ello, se emplea un archivo de configuración (por defecto, prepare_data_config.yaml), donde se definen submódulos:
yaml
CopiarEditar
symbols: [BTCUSDT, ETHUSDT]
timeframes: [1d, 4h]

indicators:
  timeframes:
    "1d":
      sma: [50, 200]
      fib_window: 90
      volatility_window: 20
    "4h":
      sma: [20, 50]
      fib_window: 45
      volatility_window: 14
  scaling_factors:
    "1d": 252
    "4h": 63

zones:
  pivot_window: 3
  volume_threshold: 1.5
  atr_multiplier: 0.5
  dbscan_eps: 0.02
  dbscan_min_samples: 1
  mecha_thr_demand: 0.2
  mecha_thr_supply: 0.2
  body_thr_demand: 0.4
  body_thr_supply: 0.4
  use_fib: true
  use_volume_delta: true
  freshness_penalty: 0.01

context:
  multi_tf_validation: true
  overlap_tolerance: 0.02
  use_key_events: false
  decay_hours: 12
  assign_zone_to_candles: true
  zone_assign_dist_mode: "closest_mean"

labeling:
  # futuro: definiciones de umbrales para rebote, breakout, etc.
  # horizon: 5
  # threshold_pct: 2.0

En el orquestador, se llama:
python
CopiarEditar
config = load_config("prepare_data_config.yaml")
indicators_cfg = config["indicators"]
zones_cfg      = config["zones"]
context_cfg    = config["context"]
labeling_cfg   = config["labeling"]  # etc.

Y cada script (indicators, zones, context, labeling) se limita a recibir la sección que le corresponde y no carga el archivo YAML directamente.
4. Descripción de Cada Módulo
4.1 indicators.py
Función principal: calculate_technical_indicators(df, timeframe, indicators_cfg).
Responsabilidad: Añadir indicadores como RSI, MACD, ATR, Bollinger, Fibonacci, volumen delta, SMAs, etc.
Config submódulo: indicators_cfg, donde se definen parámetros por timeframe (ventanas SMA, fib_window, volatility_window, scaling_factors, etc.).
Salida: Devuelve un DataFrame con las nuevas columnas de indicadores.
4.2 zones.py
Función principal: run_zones_pipeline(df, zones_cfg).
Responsabilidad:
identify_zones(): detecta pivots (swing lows/highs) usando pivot_window, volumen (volume_threshold), mecha/body ratios (mecha_thr_*, body_thr_*), y ATR.
cluster_zones(): fusiona pivots con DBSCAN (eps=dbscan_eps) para unificar zonas.
calculate_zone_strength(): genera zone_raw_strength (submodelo placeholder) usando repetitions, volume_score, freshness, fib overlap, etc.
Config submódulo: zones_cfg (p. ej. dbscan_eps, pivot_window, etc.).
Salida: (df_main_zones, df_zones_clusters), donde:
df_main_zones: conserva 1 fila por vela, con columnas demand_zone_lower, supply_zone_upper, etc.
df_zones_clusters: 1 fila por zona unificada (cluster) con zone_raw_strength.
4.3 context.py
Función principal: run_context_pipeline(df_main, df_zones, df_zones_higher, key_events, context_cfg).
Responsabilidad:
validate_cross_temporal_zones(): si multi_tf_validation=true, compara df_zones de 4H contra df_zones_higher de 1D, marcando validated_by_higher.
integrate_key_events(): placeholder para event_impact (on-chain, macro).
calculate_zone_relevance(): combina zone_raw_strength + event_impact + validated_by_higher.
assign_zones_to_candles(): (opcional) genera col closest_demand_relevance en df_main.
Config submódulo: context_cfg (p. ej. overlap_tolerance, use_key_events, assign_zone_to_candles...).
Salida: (df_main_context, df_zones_context) con relevancias finales.
4.4 labeling.py
Función principal: run_labeling_pipeline(df_main_context, labeling_cfg).
Responsabilidad: Etiquetar velas con eventos (rebote, ruptura, consolidación, etc.) según la interacción con zonas o reglas definidas.
Config submódulo: labeling_cfg (umbral de % rebote, horizonte de velas, etc.).
Salida: df_main_labeled con 1 fila por vela y col label o event_type.
4.5 logs.py
Manejo unificado de logs:
Provee log_event(level, msg, module) que escribe en un archivo prepare_data.log (y opcionalmente en consola con colores).
Todos los scripts (indicators, zones, context, labeling, main) importan log_event(...) para unificar el formato y destino.

5. Posibles Pendientes y Futuras Mejoras
Validaciones Más Exhaustivas
Revisar NaNs, duplicados, stats de # de filas tras cada paso.
Logear avisos/warnings cuando algo sea anormal.
Pruebas Unitarias
Usar pytest en una carpeta tests/, con dataset pequeños de ejemplo.
Verificar que indicators, zones, context, labeling funcionen según lo esperado.
Modo Incremental vs. Batch
Actualmente, se procesa batch. Para grandes volúmenes de datos, se podría implementar un modo incremental que solo procese la parte nueva.
Mejorar “Key Events”
Añadir on-chain o macro news, decaimiento temporal, etc. en context (use_key_events: true).
Orquestadores Avanzados
Migrar a Airflow/Prefect si se requieren DAGs complejos o programaciones automáticas.
Sub-modelos
Reemplazar pesos fijos en calculate_zone_strength() con un XGBoost entrenado históricamente para predecir la eficacia de la zona.
Labeling Avanzado
Definir horizontes de predicción (ej. “rebote +3% en las siguientes 10 velas”).
O un labeling multicategoría (rebote leve, rebote fuerte, ruptura, etc.).

6. Conclusión
El paquete prepare_data/ conforma la columna vertebral de la preparación de datos para el sistema de trading. A través de scripts modulares (indicadores, zonas, contexto, labeling), se generan datasets limpios y enriquecidos. Su uso se centraliza en main.py (orquestador), que:
Carga la config (submódulos: indicators, zones, context, labeling).
Ejecuta cada paso en secuencia, generando parquets intermedios para depurar y diagnosticar.
Produce un resultado final _labeled.parquet (cuando labeling está activo) apto para ML.
Con las mejoras pendientes (validaciones, tests, sub-modelos, key events), el pipeline puede crecer y adaptarse a requisitos más avanzados, sin perder su diseño modular. ¡Así, prepare_data/ da un salto de datos validados crudos a un dataset listo para la inteligencia artificial!

