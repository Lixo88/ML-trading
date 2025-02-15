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
pip install binance-client pandas pyarrow pyyaml
Configura el archivo config.yaml con tus parámetros.

Ejecuta el script principal:
Los datos descargados y validados estarán disponibles en la carpeta data/, organizados por símbolo.

Los datos se almacenan en formato columnar (Parquet) para eficiencia.

Se mantiene un registro detallado de todo el proceso.

Los datos están listos para la fase de feature engineering.

