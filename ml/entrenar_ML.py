import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

print("[+] Iniciando pipeline de entrenamiento híbrido...")

# 1. CARGA DE DATOS HISTÓRICOS
try:
    # Se asume un archivo histórico con columnas base de viajes o muestreos previos
    df_hist = pd.read_csv("ecobici_datos_20k.xls")
except Exception as e:
    print(f"[-] Error: No se encontró 'ecobici_datos_20k.csv'. Descárgalo del portal de datos abiertos. {e}")
    exit(1)

# Normalización estructural del csv
if 'Cicloestacion_Retiro' in df_hist.columns:
    df_hist = df_hist.rename(columns={
        'Cicloestacion_Retiro': 'station_id',
        'Fecha_Retiro': 'fecha',
        'Hora_Retiro': 'hora'
    })

# Asegurar columnas mínimas requeridas para el entrenamiento sintético/real
if 'station_id' not in df_hist.columns:
    df_hist['station_id'] = np.random.choice(range(1, 100), size=len(df_hist))
if 'capacity' not in df_hist.columns:
    df_hist['capacity'] = np.random.choice([15, 23, 27, 35, 47], size=len(df_hist))
if 'bikes_avail' not in df_hist.columns:
    df_hist['bikes_avail'] = np.random.randint(0, df_hist['capacity'] + 1)

# 2. INGENIERÍA DE CARACTERÍSTICAS TEMPORALES Y AVANZADAS
# 1) Calculamos el 'pct_full' (tasa de ocupación actual).
# 2) Ciclicidad Temporal: Pasamos la hora a funciones Seno y Coseno para que el modelo entienda la continuidad del tiempo (que las 23:00 h y las 00:00 h son consecutivas).
# 3) Lags y Series de Tiempo: Usamos '.shift()' por estación para calcular retrasos temporales, derivadas de velocidad de cambio y aceleración de la demanda.
# 4) Geolocalización: Segmentamos la CDMX en 4 regiones discretas mediante porcentajes/qcut (Norte, Centro, Sur, Poniente) .
df_hist['pct_full'] = df_hist['bikes_avail'] / df_hist['capacity'].replace({0: 1})

if 'fecha' in df_hist.columns:
    df_hist['fecha_dt'] = pd.to_datetime(df_hist['fecha'], errors='coerce')
    df_hist['hora'] = df_hist['fecha_dt'].dt.hour
    df_hist['dia_semana'] = df_hist['fecha_dt'].dt.dayofweek
else:
    df_hist['hora'] = np.random.randint(0, 24, size=len(df_hist))
    df_hist['dia_semana'] = np.random.randint(0, 7, size=len(df_hist))

df_hist['es_fin_de_semana'] = df_hist['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
df_hist['hora_sin'] = np.sin(2 * np.pi * df_hist['hora'] / 24.0)
df_hist['hora_cos'] = np.cos(2 * np.pi * df_hist['hora'] / 24.0)
df_hist['hora_tipo_dia'] = df_hist['hora'] + (df_hist['es_fin_de_semana'] * 24)

# Creación de Lags y Series de Tiempo Estáticas para entrenamiento
df_hist['pct_full_log1'] = df_hist.groupby('station_id')['pct_full'].shift(1).fillna(df_hist['pct_full'])
df_hist['pct_full_log2'] = df_hist.groupby('station_id')['pct_full'].shift(2).fillna(df_hist['pct_full'])
df_hist['pct_full_log3'] = df_hist.groupby('station_id')['pct_full'].shift(3).fillna(df_hist['pct_full'])
df_hist['pct_full_log_1h'] = df_hist.groupby('station_id')['pct_full'].shift(4).fillna(df_hist['pct_full'])
df_hist['pct_full_log_2h'] = df_hist.groupby('station_id')['pct_full'].shift(8).fillna(df_hist['pct_full'])

df_hist['velocidad_cambio'] = df_hist['pct_full_log1'] - df_hist['pct_full_log2']
df_hist['aceleracion_cambio'] = df_hist['pct_full_log1'] - (2 * df_hist['pct_full_log2']) + df_hist['pct_full_log3']
df_hist['tendencia_1h'] = df_hist['pct_full'] - df_hist['pct_full_log_1h']
df_hist['aceleracion_tendencia_1h'] = df_hist['pct_full'] - (2 * df_hist['pct_full_log_1h']) + df_hist['pct_full_log_2h']

# Asignación de Coordenadas Geográficas Base (Área Metropolitana CDMX)
if 'lat' not in df_hist.columns:
    df_hist['lat'] = np.random.uniform(19.350, 19.450, size=len(df_hist))
    df_hist['lon'] = np.random.uniform(-99.200, -99.130, size=len(df_hist))

# Clusterización Logística Rápida (Zonas Norte, Centro, Sur, Poniente)
df_hist['zona_logistica'] = pd.qcut(df_hist['lat'], q=4, labels=[1, 2, 3, 4]).astype(int)

# 3. EXTRACCIÓN DE CONOCIMIENTO HISTÓRICO
# 1) Exportamos los mapas de comportamiento histórico y codificaciones a archivos CSV. 
#    El script de tiempo real usará estos archivos para cruzar la información 
#    en vivo sin tener que recalcular bases de datos masivas constantemente.
# 2) Construcción de Targets: Desfasamos negativamente (-2 posiciones) la disponibilidad 
#    para crear la variable objetivo del regresor. Posteriormente, aplicamos la regla estricta 
#    del semáforo de 4 clases basado en la capacidad para construir el target del clasificador.
mapa_historico = df_hist.groupby(['station_id', 'hora']).agg(
    rango_historico_estacion=('pct_full', lambda x: x.max() - x.min())
).reset_index()

encoding_map = df_hist.groupby(['station_id', 'hora_tipo_dia']).agg(
    ocupacion_historica_estacion_hora=('pct_full', 'mean')
).reset_index()

coords_base = df_hist[['lat', 'lon', 'zona_logistica']].drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)

mapa_historico.to_csv("mapa_historico.csv", index=False)
encoding_map.to_csv("encoding_map.csv", index=False)
coords_base.to_csv("coords_base.csv", index=False)

df_hist = df_hist.merge(mapa_historico, on=['station_id', 'hora'], how='left')
df_hist = df_hist.merge(encoding_map, on=['station_id', 'hora_tipo_dia'], how='left')

# Definición del Target del Regresor: cuántas bicicletas habrá en el futuro (ej. +2 horas)
df_hist['target_bikes_2h'] = df_hist.groupby('station_id')['bikes_avail'].shift(-2).fillna(df_hist['bikes_avail'])

# Definición del Target del Clasificador (Semáforo de Criticidad Cruda)
# 0: Vacía/Crítica, 1: Normal-Baja, 2: Normal-Alta, 3: Llena/Saturada
def categorizar_estado(row):
    pct = row['target_bikes_2h'] / max(1, row['capacity'])
    if pct <= 0.15: return 0
    elif pct <= 0.42: return 1
    elif pct <= 0.85: return 2
    else: return 3

df_hist['target_estado_2h'] = df_hist.apply(categorizar_estado, axis=1)

# 4. ENTRENAMIENTO DE MODELOS XGBOOST
# 1) Primero, el XGBRegressor se entrena con las 20 características base para predecir la cantidad exacta de bicicletas a futuro ('target_bikes_2h').
# 2) Inyección de Meta-Feature: Ejecutamos '.predict()' con el regresor entrenado sobre los mismos datos y guardamos el resultado continuo en la columna 'pred_num_regresor'.
# 3) Esta predicción se suma al vector de características original.
# 4) Finalmente, el XGBClassifier se entrena utilizando esta variable matemática para aprender a predecir la clasificación correcta del semáforo en un entorno no lineal.
# Ambos modelos se exportan en formato JSON liviano de producción.
FEATURES_REGRESOR = [
    'capacity', 'bikes_avail', 'pct_full', 'hora_sin', 'hora_cos', 'dia_semana', 
    'es_fin_de_semana', 'hora_tipo_dia', 'pct_full_log1', 'pct_full_log2', 'pct_full_log3',
    'pct_full_log_1h', 'pct_full_log_2h', 'rango_historico_estacion', 'velocidad_cambio', 
    'aceleracion_cambio', 'tendencia_1h', 'aceleracion_tendencia_1h', 'ocupacion_historica_estacion_hora', 'zona_logistica'
]


regresor = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
regresor.fit(df_hist[FEATURES_REGRESOR], df_hist['target_bikes_2h'])
regresor.save_model("regresor_ecobici.json")

df_hist['pred_num_regresor'] = regresor.predict(df_hist[FEATURES_REGRESOR])
FEATURES_CLASIFICADOR = FEATURES_REGRESOR + ['pred_num_regresor']

clasificador = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
clasificador.fit(df_hist[FEATURES_CLASIFICADOR], df_hist['target_estado_2h'])
clasificador.save_model("clasificador_ecobici.json")

print("[+] ¡Proceso completado exitosamente!")