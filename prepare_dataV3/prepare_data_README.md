# `prepare_data/` — Data Pipeline for ML Trading

## Objetivos Generales

Este paquete implementa el **pipeline de preparación de datos** para un sistema de trading algorítmico basado en supply/demand zones y un modelo de Machine Learning (e.g. XGBoost). Los pasos fundamentales son:

1. **Descarga & Validación** (hecho en `download_data/`, resultando en `*_validated.parquet`).
2. **Indicadores Técnicos** (rsi, macd, atr, adx, etc.) → `indicators.py`.
3. **Detección de Zonas** (demand/supply) → `zones.py`.
4. **Análisis de Contexto** (multi-timeframe, key events, zone relevance) → `context.py`.
5. **Labeling** ( merges multi-TF, feature engineering final, creación de etiqueta rebote/rompimiento ) → `labeling.py`.
6. **Orquestación** (archivos intermedios, logging unificado) → `main.py` (o `__main__.py`).

El resultado es uno o varios **.parquet** finales que contienen features + label, aptos para entrenar un modelo ML (XGBoost).

## Estructura del Paquete

prepare_data/ 
├── init.py 
├── main.py # Orquestador global, genera _with_indicators, _with_zones, _with_context 
├── indicators.py # Cálculo no-leaking de indicadores (RSI, MACD, ADX, etc.) 
├── zones.py # Detección adaptativa de zonas (pivot, mecha/body, DBSCAN cluster) 
├── context.py # Validación multi-TF, zone relevance, asignación de zona a velas 
├── labeling.py # Unificación multi-TF, feature engineering final, etiquetado (rebote/breakout) 
├── logs.py # Logging estructurado JSON + colores en consola  
└── prepare_data_config.yaml # Config YAML global (opcional)

**Flujo Lógico**:  
1. `main.py` → para cada `(symbol, timeframe)`, llama:
   - `indicators.py` => `_with_indicators.parquet`
   - `zones.py` => `_with_zones.parquet` + `_zones_clusters.parquet`
   - `context.py` => `_with_context.parquet` + `_zones_context.parquet`
2. Luego, `labeling.py` produce un dataset final “`symbol_startdate-enddate_prepared.parquet`” que contiene merges multiTF + etiquetado de rebote/rompimiento.

## Estrategia y Fases

1. **Indicators**: Se calculan con `shift(1)` o `shift(2)` para evitar leakage (mirando solo datos pasados). Se generan docenas de TA (RSI, MACD_line, ATR, ADX, Bollinger, Fib, etc.).
2. **Zonas**: Se detectan pivots adaptativos con ATR + mecha/body ratio y volumen (zscore o relative). Se aplican transformaciones (Power/Quantile) y DBSCAN en un espacio multidimensional. Se obtiene un `zone_strength` con ranking percentil y placeholders para submodelos.
3. **Context**:  
   - Valida multi-timeframe (4H vs 1D) con `IntervalTree`, añade `validated_by_higher`, `event_impact`, etc.  
   - Calcula `zone_relevance`.  
   - Asigna la zona más fuerte a cada vela (`closest_demand_relevance`, `closest_supply_relevance`).
4. **Labeling**:  
   - Fusiona la TF base (4H) con otras TF (1D, 1H, 15m) para features.  
   - Crea la etiqueta final (rebote=1, rompimiento=2, etc.) con `label_events()`.  
   - Genera un dataset “`symbol_startdate-enddate_prepared.parquet`” apto para ML.

## Módulos:

### `indicators.py`

**Objetivo**: Calcular un amplio conjunto de features técnicos de manera **no-leaking**, integrando indicadores clásicos y técnicas avanzadas de normalización y market phase detection.

**Características Clave**:
1. **Prevención de Data Leakage**:
   - `shift(2)` en `returns` para no revelar información de la vela actual o la vela inmediata anterior.
   - `shift(1)` en RSI, MACD, ADX, Bollinger, Fibonacci, SMAs para que cada valor en la vela t solo use datos hasta la vela t-1.

2. **Market Phase** con ADX:
   - Determina si el mercado está en tendencia alcista, bajista o lateral según `adx_threshold` y `di_plus/di_minus`.

3. **Volumen Inteligente**:
   - Calcula `buy_vol_ema` y `sell_vol_ema` con un `ewm(span=14)`, y su ratio `volume_ratio`.
   - Evita división por cero.

4. **Indicadores Tradicionales**:
   - Bollinger Bands (`bb_upper`, `bb_lower`, `bb_ratio`),
   - ATR y ADX (`adx`, `di_plus`, `di_minus`),
   - Fibonacci Rolling (dist_to_fib_0.236, etc.),
   - SMAs dinámicas (parámetros en `sma_periods`).

5. **Normalización**:
   - `close_norm = close_t1 / sma_200(t1)` para comparar activos distintos.

6. **Volatilidad**:
   - `volatility = rolling_std(returns, window=volatility_window) * sqrt(scale)`.

**Configuración**:
```yaml
indicators:
  timeframes:
    "4h":
      sma: [20, 50]
      fib_window: 45
      volatility_window: 14
      adx_threshold: 25
  scaling_factors:
    "4h": 63
    ```

**Notas:**

    Si la posterior fase ML requiere no ver el returns[t], se puede shift(1) adicional a la label. La actual approach añade robustez a la pipeline.
    adx_threshold, ewm(span=14), etc. son arbitrarios y podrían optimizarse con un sub-modelo.
    Salida:

    Un DataFrame con ~30+ columnas de features, sin leakage inmediato, y log detallado para debugging.

### `zones.py`

**Objetivo**: Detectar y unificar zonas de oferta/demanda en base a pivots, volumen y clusterización avanzada, generando un dataframe de zonas con su fuerza.

**Características Principales**:

1. **Pivots Adaptativos** con ATR:
   - Define `dynamic_window = ~ 5 * (ATR actual / ATR histórico)`, saturado entre `[pivot_window, 10]`.
   - Filtra mecha/body ratio para demand/supply (p. ej., `mecha_thr_demand=0.2`).

2. **Volumen Parametrizable**:
   - `volume_approach: "zscore"` => filtra pivots con `volume_zscore >= threshold`.
   - `"relative"` => filtra con `relative_volume >= threshold`.

3. **Clustering Multidimensional**:
   - Transforma (precio log, tiempo en horas, volumen / zscore) con `PowerTransformer`, `QuantileTransformer`.
   - DBSCAN adaptativo (`min_samples_factor = 0.05`) => >=5% de pivots conforman un cluster.
   - Devuelve un `df_zones` con `[zone_lower, zone_upper, pivot_mean_price, zone_freshness, etc.]`.

4. **Cálculo de Strength**:
   - Ranking percentil de `repetitions`, `volume_score`, `zone_freshness`.
   - Combina con pesos (`strength_weights`) y normaliza en [0..1].
   - Placeholders para submodelo ML (p. ej. XGBoost) que podría reemplazar el linear/ranking approach.

5. **Salida**:
   1. `df_main`: conserva 1 fila por vela y añade `demand_zone_lower/upper`, `supply_zone_lower/upper`.
   2. `df_zones`: 1 fila por cluster unificado, con `zone_strength`.

**Configuración**:
```yaml
zones:
  pivot_window: 3
  atr_multiplier: 1.5
  volume_approach: "zscore"
  volume_threshold: 1.5
  volume_zscore_threshold: 2.0
  mecha_thr_demand: 0.2
  mecha_thr_supply: 0.2
  body_thr_demand: 0.0
  body_thr_supply: 0.0
  dbscan_eps: 0.02
  min_samples_factor: 0.05
  repetition_window: 90
  freshness_penalty: 0.01
  strength_weights: [0.4, 0.3, 0.2, 0.1]
  use_power_transform: true
  use_quantile_transform: true
  use_fib: true
  use_volume_delta: true
  ```

**Posibles Parámetros para un Submodelo:**

    volume_zscore vs. relative_volume
    body_ratio, lower_wick_ratio, upper_wick_ratio
    zone_freshness, repetitions, volume_score, pivot_mean_price
    Los pesos en strength_weights podrían provenir de un XGBoost entrenado con histórico (e.g. “prob de rebote con +2%”).

    ### `context.py`

**Objetivo**: Integrar un *contexto multi-timeframe* y/o *eventos clave*, calcular la relevancia final de las zonas, y asignar la zona más fuerte a cada vela.

**Características Principales**:

1. **Validación Multi-TF** con `IntervalTree`:
   - `df_zones_higher` se construye en la temporalidad superior (p.ej. 1D), se añade al tree.
   - Marca `validated_by_higher = True` si la zona en TF actual solapa con la zona superior (± overlap_tolerance).

2. **Integración de Key Events**:
   - `event_impact` = 0.0 por defecto (placeholder).
   - En un futuro, decaimiento exponencial u otra lógica para “FOMC events”, “whale moves”, etc.

3. **Liquidity Profile** (Opcional):
   - `use_liquidity_profile`: calcula `liquidity_profile` en `df_main` con una rolling-apply, normalizando por ATR, etc.
   - Factor extra si deseas integrarlo en “zone_relevance.”

4. **Cálculo Final de Relevancia**:
   ```python
   zone_relevance = (strength_weight * zone_strength
                     + validation_weight * validated_by_higher
                     + event_impact_weight * event_impact)

    Placeholder submodelo: un XGBoost podría reemplazarlo y usar liquidity_profile, etc.

5. **Asignación de Zonas a Velas:**
Crea dos IntervalTrees (demand vs supply).
Para cada vela, busca overlap([low, high]) y toma la zona con mayor zone_relevance.
Genera closest_demand_relevance / closest_supply_relevance en df_main.

6. **Salida:**
df_main_context: candle-based con liquidity_profile, closest_demand_relevance, closest_supply_relevance (si config dice).
df_zones_context: zone-based con zone_relevance, validated_by_higher, event_impact.

**Configuración (ejemplo):**
```yaml
context:
  multi_tf_validation: true
  overlap_tolerance: 0.02
  use_key_events: false
  event_impact_weight: 0.3
  use_liquidity_profile: true
  liquidity_window: 20
  price_tolerance: 0.02
  strength_weight: 1.0
  validation_weight: 1.0
  assign_zone_to_candles: true
  ```

**Parámetros Refinables (Submodelo):**
    strength_weight, validation_weight, event_impact_weight => podrían aprenderse de forma automática en un ML si dispones de labels de “zona exitosa.”
    liquidity_window, price_tolerance => calibrar en backtesting.
    overlap_tolerance => define la relajación en la validación multiTF.

### `labeling.py`

**Objetivo**: Generar un **solo dataset final** con (a) features unificados de múltiples TF, (b) la información de zones/context, y (c) la **etiqueta** de evento (rebote, rompimiento, etc.), quedando listo para entrenar un modelo supervisado (XGBoost, etc.).

**Puntos Clave**:

1. **Carga**:
   - Usa `_with_context.parquet` para la TF principal (p.ej. 4H) y para las otras TFs (1D, 1H, 15m).
2. **Fusión** Multi-TF:
   - `enrich_with_higher_tf(...)` => une 1D con 4H (`merge_asof`, resample).  
   - `aggregate_lower_tf(...)` => agrega 1H, 15m en ventanas de 4H (volumen sum, zone_relevance max, etc.).
3. **Feature Engineering** Adicional:
   - Crea `returns_main, volatility_main, zone_impact, macro_micro_signal`, etc. 
   - Se dejan placeholders en el YAML para calibrar parámetros.
4. **Etiquetado** de rebote / rompimiento / falso breakout:
   - Revisar proximas N velas tras tocar una zona y ver si sube +X%, rompe, etc.
   - `df['label'] = ... (0=none,1=rebote,2=breakout,...)`.
5. **Salida**:
   - `symbol_startdate-enddate_prepared.parquet` con ~50+ columnas (TA, zone, multiTF merges) y la `label`.
   - Listo para `XGBoostClassifier(...)`.

**Configuración**:
```yaml
labeling:
  main_timeframe: "4h"
  higher_tf: "1d"
  lower_tfs: ["1h","15m"]
  rebote_window: 3
  rebote_threshold: 0.015
  breakout_window: 3
  breakout_threshold: 0.01
  ...
  output_dir: "data_ml"
  ```

**Parámetros Entrenables vs. Submodelo:**
    En YAML:
    rebote_window, rebote_threshold = definiciones de la lógica de etiquetado.
    merges y aggregator approach (vol sum, zone_relevance max, etc.).
    En el Submodelo (XGBoost):
    Pesos de RSI, zone_impact, liquidity, etc. se aprenden al entrenar.
    Se evita “hard-codear” su importancia.

### `logs.py`

**Objetivo**: Proveer un **logger centralizado** que:

1. **Escribe** un archivo de log en formato **JSON** (con campos `timestamp, level, module, message, metadata`).
2. **Muestra** en consola mensajes con color (usando colorama) para niveles INFO, WARNING, ERROR, etc.
3. **Funciona** con funciones de alto nivel:
   - `log_event(level, msg, module, metadata={})`
   - `log_feature_stats(df, features, module)`

**Características**:
- **Archivo**: `logs/pipeline_YYYYMMDD.log`, con una línea JSON por evento.  
- **Consola**: Color-coded (verde=INFO, amarillo=WARNING, rojo=ERROR, azul=DEBUG).  
- **Metadata**: Se le pasa un diccionario con info extra (p.ej. `"symbol": "BTCUSDT"`).  
- **Estadísticas**: `log_feature_stats(df, ["rsi","macd"], "indicators")` produce un log con mean, std, etc. en `metadata.stats`.

**Ejemplo**:
```python
from prepare_data.logs import log_event, log_feature_stats
import pandas as pd

df = pd.DataFrame({"rsi":[30,50,70], "macd":[-0.1,0.05,0.2]})
log_event("INFO", "Cálculo de RSI completado", module="indicators", metadata={"symbol":"BTCUSDT"})
log_feature_stats(df, ["rsi","macd"], module="indicators")

### `main.py`

**Objetivo**: Orquestar todo el pipeline de preparación de datos. Para cada `(symbol, timeframe)`:

1. Busca `*_validated.parquet` en `data_processed/<symbol>/`.
2. Llama:
   - **indicators** => `_with_indicators.parquet`
   - **zones** => `_with_zones.parquet` / `_zones_clusters.parquet`
   - **context** => `_with_context.parquet` / `_zones_context.parquet`
3. Guarda salidas en `data_prepared/indicators`, `.../zones`, `.../context`.
4. Usa **logging** unificado con `log_event(...)`.
5. Multi-símbolo, multi-timeframe, con **paralelización** (joblib).

**Uso**:
```bash
python main.py --config prepare_data_config.yaml \
               --data_dir data_processed \
               --output_dir data_prepared \
               --symbols BTCUSDT ETHUSDT
```

**Parámetros Principales (en prepare_data_config.yaml):**
```yaml
symbols: [BTCUSDT, ETHUSDT]
timeframes: [1d, 4h, 1h, 15m]
```
**Salida:**
data_prepared/indicators/SYMBOL_4h_..._with_indicators.parquet
data_prepared/zones/SYMBOL_4h_..._with_zones.parquet
data_prepared/context/SYMBOL_4h_..._with_context.parquet


---

## Conclusión

Este `main.py` unificado, junto a la configuración YAML y el pipeline de submódulos (*indicators*, *zones*, *context*), completa el **flujo**:

1. **Carga** config y busca `_validated.parquet`.
2. **Calcula** indicadores → zona → contexto.
3. **Guarda** resultados intermedios con `_with_indicators`, `_with_zones`, `_with_context`.
4. **Soporta** varios símbolos/timeframes, con logs y CLI flexible.

Así, el pipeline de “prepare_data/” está **listo** para la fase final (`labeling.py`) que generará un dataset unificado con features + label. ¡Éxito en la implementación!
