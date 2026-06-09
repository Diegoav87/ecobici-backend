import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests
import warnings
import time

warnings.filterwarnings('ignore')

URL_STATION_INFO = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_information.json"
URL_STATION_STATUS = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_status.json"
DISTANCIA_LIMITE_GRADOS = 0.009

# LISTA DE VARIABLES
FEATURES_REGRESOR = [
    'capacity', 'bikes_avail', 'pct_full', 'hora_sin', 'hora_cos', 'dia_semana', 
    'es_fin_de_semana', 'hora_tipo_dia', 'es_pico_semana', 'pct_full_log1', 'pct_full_log2', 'pct_full_log3',
    'pct_full_log_1h', 'rango_historico_estacion', 'velocidad_cambio', 
    'aceleracion_cambio', 'tendencia_1h', 'ocupacion_historica_estacion_hora', 
    'slots_vacios', 'zona_logistica', 'rolling_mean_30m', 'rolling_std_30m', 'rolling_mean_1h',
    'interaccion_cap_hora', 'historico_estacion_hora_dia', 'desviacion_de_media_30m', 'momentum_quiebre_30m'
]

# 1. CARGA DE MODELOS PRE-ENTRENADOS
try:
    # Modelo Regresor: Estima el volumen numérico exacto de bicicletas que habrá en 50 minutos
    model_regresor = xgb.XGBRegressor()
    model_regresor.load_model("regresor_ecobici.json")

    # Modelo Clasificador: Recibe los features más la predicción del regresor para categorizar la gravedad del estado de la estación
    model_clasificador = xgb.XGBClassifier()
    model_clasificador.load_model("clasificador_ecobici.json")

    # DataFrames estáticos generados en el script de entrenamiento para inyectar patrones históricos de uso (Target Encoding)
    mapa_tri_interaccion = pd.read_csv("mapa_tri_interaccion.csv") # Combinación Estación + Hora + Día de la semana
    mapa_historico = pd.read_csv("mapa_historico.csv") # Combinación general Estación + Hora
    encoding_map = pd.read_csv("encoding_map.csv") # Clasificación esperada según estación y tipo de día
    coords_base = pd.read_csv("coords_base.csv") # Mapeo pre-calculado de coordenadas a clústeres logísticos
except Exception as e:
    print(f"[-] Error crítico al cargar entorno: {e}")
    exit(1)

# BUFFER EN MEMORIA
# Dado que la API comercial limpia sus estados pasados en segundos (TTL corto), creamos una base de datos temporal
# local en memoria RAM para simular las funciones .shift() del entrenamiento sin recurrir a consultas de base
historico_estaciones = {}  
MAX_MINUTOS_BUFFER = 70

while True:
    try:
        # 2. INGESTA EN VIVO DE DATOS
        res_info = requests.get(URL_STATION_INFO, timeout=15).json()
        res_disp = requests.get(URL_STATION_STATUS, timeout=15).json()

        # Normalización estructural de JSON a DataFrames estructurados indexables
        df_info_api = pd.DataFrame([{'station_id': int(s['station_id']), 'lat': float(s['lat']), 'lon': float(s['lon']), 'capacity': int(s['capacity'])} for s in res_info['data']['stations']])
        df_disp_api = pd.DataFrame([{'station_id': int(s['station_id']), 'bikes_avail': int(s['num_bikes_available']), 'is_renting': int(s['is_renting']), 'is_returning': int(s['is_returning'])} for s in res_disp['data']['stations']])

        # Cruce interno (Inner Join) mediante el identificador único de estación para consolidar variables espaciales y dinámicas
        df_realtime = pd.merge(df_disp_api, df_info_api, on='station_id', how='inner')
        df_realtime['pct_full'] = df_realtime['bikes_avail'] / df_realtime['capacity'].replace({0: 1})
        df_realtime['slots_vacios'] = df_realtime['capacity'] - df_realtime['bikes_avail']

        # Filtro de funcionamiento operativo de la estación
        df_realtime = df_realtime[(df_realtime['is_renting'] == 1) & (df_realtime['is_returning'] == 1)].reset_index(drop=True)

        # 3. VARIABLES TEMPORALES Y ENRIQUECIMIENTO
        # Mantener la consistencia de hora y día en todo el procesamiento.
        ts_reciente = pd.Timestamp.now()
        df_realtime['hora'] = ts_reciente.hour
        df_realtime['dia_semana'] = ts_reciente.dayofweek
        df_realtime['es_fin_de_semana'] = df_realtime['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

        # Transformación matemática de la hora a coordenadas polares (Seno/Coseno) para que el modelo reconozca la continuidad temporal entre las 23:59 y las 00:00
        df_realtime['hora_sin'] = np.sin(2 * np.pi * df_realtime['hora'] / 24.0)
        df_realtime['hora_cos'] = np.cos(2 * np.pi * df_realtime['hora'] / 24.0)
        
        df_realtime['hora_tipo_dia'] = df_realtime['hora'] + (df_realtime['es_fin_de_semana'] * 24)
        df_realtime['es_pico_semana'] = df_realtime['dia_semana'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        df_realtime['interaccion_cap_hora'] = df_realtime['capacity'] * df_realtime['hora_cos']

        # Mapeo de promedios históricos en base a registros pasados
        df_realtime = df_realtime.merge(mapa_tri_interaccion, on=['station_id', 'hora', 'dia_semana'], how='left')
        df_realtime = df_realtime.merge(mapa_historico, on=['station_id', 'hora'], how='left')
        df_realtime = df_realtime.merge(encoding_map, on=['station_id', 'hora_tipo_dia'], how='left')
        df_realtime = df_realtime.merge(coords_base, on=['lat', 'lon'], how='left')
        
        # Reconstrucción de rezagos (lags) y rolling metrics temporales a partir de la memoria simulada.
        list_log1, list_log2, list_log3, list_log_1h = [], [], [], []
        list_roll_mean_30, list_roll_std_30, list_roll_mean_1h = [], [], []

        for idx, row in df_realtime.iterrows():
            s_id = row['station_id']
            current_pct = row['pct_full']
            
            if s_id not in historico_estaciones:
                historico_estaciones[s_id] = []
            
            # Guardar la foto del estado actual en el buffer de la estación
            historico_estaciones[s_id].append((ts_reciente, current_pct))
            
            # Limpieza de registros
            limite_tiempo = ts_reciente - pd.Timedelta(minutes=MAX_MINUTOS_BUFFER)
            historico_estaciones[s_id] = [h for h in historico_estaciones[s_id] if h[0] > limite_tiempo]
            
            # Busqueda eventos pasados exactos dentro de las ventanas móviles
            def buscar_en_pasado(minutos_atras):
                target_ts = ts_reciente - pd.Timedelta(minutes=minutos_atras)
                for ts_hist, pct_hist in reversed(historico_estaciones[s_id]):
                    if abs((ts_hist - target_ts).total_seconds()) <= 120:
                        return pct_hist
                return current_pct

            # Extracción exacta de estados pasados basados en pasos operativos
            pct_log1 = buscar_en_pasado(10)
            pct_log2 = buscar_en_pasado(20)
            pct_log3 = buscar_en_pasado(30)
            pct_log_1h = buscar_en_pasado(60)

            list_log1.append(pct_log1)
            list_log2.append(pct_log2)
            list_log3.append(pct_log3)
            list_log_1h.append(pct_log_1h)
            
            # Generación de ventanas dinamicas
            arr_30m = np.array([pct_log1, pct_log2, pct_log3])
            arr_1h = np.array([pct_log1, pct_log2, pct_log3, pct_log_1h])

            list_roll_mean_30.append(arr_30m.mean())
            list_roll_std_30.append(arr_30m.std() if len(arr_30m) > 1 else 0.0)
            list_roll_mean_1h.append(arr_1h.mean())

        # Consolidación final de la inercia temporal en el DataFrame
        df_realtime['pct_full_log1'] = list_log1
        df_realtime['pct_full_log2'] = list_log2
        df_realtime['pct_full_log3'] = list_log3
        df_realtime['pct_full_log_1h'] = list_log_1h
        
        df_realtime['rolling_mean_30m'] = list_roll_mean_30
        df_realtime['rolling_std_30m'] = list_roll_std_30
        df_realtime['rolling_mean_1h'] = list_roll_mean_1h

        # CÁLCULO DE DIFERENCIALES DINÁMICOS
        df_realtime['velocidad_cambio'] = df_realtime['pct_full_log1'] - df_realtime['pct_full_log2']
        df_realtime['aceleracion_cambio'] = df_realtime['pct_full_log1'] - (2 * df_realtime['pct_full_log2']) + df_realtime['pct_full_log3']
        df_realtime['tendencia_1h'] = df_realtime['pct_full'] - df_realtime['pct_full_log_1h']
        df_realtime['desviacion_de_media_30m'] = df_realtime['pct_full'] - df_realtime['rolling_mean_30m']
        df_realtime['momentum_quiebre_30m'] = df_realtime['velocidad_cambio'] - df_realtime['rolling_std_30m']

        df_realtime[FEATURES_REGRESOR] = df_realtime[FEATURES_REGRESOR].fillna(0)

        # 4. INFERENCIA HÍBRIDA EN VIVO
        # PASO A: El regresor estima cuántas bicicletas físicas tendrá la estación al final del horizonte predictivo (+50 min)
        df_realtime['pred_num_regresor'] = np.clip(np.round(model_regresor.predict(df_realtime[FEATURES_REGRESOR])), 0, df_realtime['capacity'])
        
        # PASO B: El clasificador recibe la predicción numérica del regresor como un feature adicional para categorizar semáforos de riesgo con mayor precisión
        features_clasificador = FEATURES_REGRESOR + ['pred_num_regresor']
        df_realtime['estado_predicho'] = model_clasificador.predict(df_realtime[features_clasificador])

        df_realtime['bicis_predichas_45m'] = df_realtime['pred_num_regresor']
        df_realtime['inventario_optimo'] = np.round(df_realtime['capacity'] * 0.425)

        # PASO C: Función de traducción de alertas a necesidades de transporte
        def calcular_unidades_netas(row):
            if row['estado_predicho'] == 0: 
                return int(max(1, row['inventario_optimo'] - row['bicis_predichas_45m']))
            elif row['estado_predicho'] == 3: 
                return int(-max(1, row['bicis_predichas_45m'] - row['inventario_optimo']))
            return 0

        df_realtime['unidades_requeridas'] = df_realtime.apply(calcular_unidades_netas, axis=1)

        # 5. ALGORITMO DE RUTEO DE RESPUESTA INMEDIATA
        # El algoritmo prioriza resolver la crisis localmente antes de delegar tareas al Almacén Central.
        def calcular_distancia_directa(lat1, lon1, lat2, lon2):
            return abs(lat1 - lat2) + abs(lon1 - lon2)

        # Regla de asignación inteligente de tipos de unidades según el volumen neto de la orden de transferencia
        def asignar_vehiculo(cantidad):
            if cantidad > 32: return "Camión Grande (Cap 50)"
            elif cantidad > 16: return "Camioneta + Remolque (Cap 32)"
            else: return "Camioneta Ligera (Cap 16 - Sin Remolque)"

        # Filtrar exclusivamente los puntos críticos que requieren movimientos físicos urgentes de unidades de transporte
        df_despacho_instante = df_realtime[df_realtime['unidades_requeridas'] != 0].copy()
        rutas_generadas = []

        if not df_despacho_instante.empty:
            for zona, grupo in df_despacho_instante.groupby('zona_logistica'):
                estaciones_oferta = grupo[grupo['unidades_requeridas'] < 0].copy()
                estaciones_demanda = grupo[grupo['unidades_requeridas'] > 0].copy()
                
                oferta_list = estaciones_oferta[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')
                demanda_list = estaciones_demanda[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')
                
                for o in oferta_list: o['disponibles'] = abs(o['unidades_requeridas'])
                for d in demanda_list: d['necesitadas'] = d['unidades_requeridas']
                
                # PASO A: Mitigación Ecológica Local (Emparejar Origen-Destino dentro de la misma zona dentro de un radio de ~1km)
                for d in demanda_list:
                    while d['necesitadas'] > 0:
                        ofertas_validas = [o for o in oferta_list if int(o['station_id']) != int(d['station_id']) and o['disponibles'] > 0]
                        if not ofertas_validas: break 
                            
                        for o in ofertas_validas:
                            o['distancia'] = calcular_distancia_directa(d['lat'], d['lon'], o['lat'], o['lon'])
                        
                        ofertas_dentro_del_radio = [o for o in ofertas_validas if o['distancia'] <= DISTANCIA_LIMITE_GRADOS]
                        if not ofertas_dentro_del_radio: break 
                        
                        # Ordenar de la estación de oferta más cercana a la más lejana
                        ofertas_dentro_del_radio = sorted(ofertas_dentro_del_radio, key=lambda x: x['distancia'])
                        mas_cercana = ofertas_dentro_del_radio[0]
                        cantidad_a_transferir = min(d['necesitadas'], mas_cercana['disponibles'])
                        
                        if cantidad_a_transferir > 0:
                            distancia_estimada_km = mas_cercana['distancia'] * 111.0
                            rutas_generadas.append({
                                'Zona Logística': int(zona), 'Estación Origen (VACIAR)': int(mas_cercana['station_id']),
                                'Estación Destino (LLENAR)': int(d['station_id']), 'Bicicletas a Mover': int(cantidad_a_transferir),
                                'Distancia (Km)': round(distancia_estimada_km, 2), 'Vehículo Asignado': asignar_vehiculo(cantidad_a_transferir)
                            })
                            d['necesitadas'] -= cantidad_a_transferir
                            for o in oferta_list:
                                if o['station_id'] == mas_cercana['station_id']: o['disponibles'] -= cantidad_a_transferir

                # PASO B: Si a la estación de demanda le siguen faltando bicicletas y ya no hay oferta en su zona, se surte del Almacén Central
                for d in demanda_list:
                    if d['necesitadas'] > 0:
                        rutas_generadas.append({
                            'Zona Logística': int(zona), 'Estación Origen (VACIAR)': "Almacén Central", 'Estación Destino (LLENAR)': int(d['station_id']),
                            'Bicicletas a Mover': int(d['necesitadas']), 'Distancia (Km)': None, 'Vehículo Asignado': asignar_vehiculo(d['necesitadas'])
                        })
                # PASO C: Si las estaciones de oferta aún tienen exceso y las estaciones de demanda locales ya se saciaron, el excedente se manda a resguardo al Almacén Central
                for o in oferta_list:
                    if o['disponibles'] > 0:
                        rutas_generadas.append({
                            'Zona Logística': int(zona), 'Estación Origen (VACIAR)': int(o['station_id']), 'Estación Destino (LLENAR)': "Almacén Central",
                            'Bicicletas a Mover': int(o['disponibles']), 'Distancia (Km)': None, 'Vehículo Asignado': asignar_vehiculo(o['disponibles'])
                        })

        df_rutas = pd.DataFrame(rutas_generadas)

        # 6. OUTPUT JSON DISPATCH PAYLOAD
        # Construcción automática de métricas operativas clave para la toma de decisiones directivas.
        df_criticas = df_realtime[df_realtime['estado_predicho'].isin([0, 3])].copy()
        total_compensado, total_necesitado = 0, 0
        
        if not df_criticas.empty:
            for zona, grupo in df_criticas.groupby('zona_logistica'):
                demand_llenar = grupo[grupo['unidades_requeridas'] > 0]['unidades_requeridas'].sum()
                total_compensado += min(demand_llenar, abs(grupo[grupo['unidades_requeridas'] < 0]['unidades_requeridas'].sum()))
                total_necesitado += demand_llenar
        
        # Sumatoria de la distancia total recorrida exclusivamente en rebalanceo de proximidad local
        distancia_local = df_rutas['Distancia (Km)'].dropna().sum() if not df_rutas.empty else 0.0

        # Compilación del objeto JSON con métricas de performance
        metricas = {
            "model_performance": {"accuracy_semaforo_pct": 81.46, "mae_volumen_bicicletas": 2.49}, 
            "green_logistics": {
                "movimientos_mitigados_unidades": int(total_compensado),
                "eficiencia_rebalanceo_local_pct": round((total_compensado / total_necesitado * 100), 2) if total_necesitado > 0 else 0.0,
                "distancia_total_optimizada_local_km": round(distancia_local, 2)
            },
            "flota_resumen": df_rutas['Vehículo Asignado'].value_counts().to_dict() if not df_rutas.empty else {}
        }

        # Estructuración final de la hoja de ruta dinámica
        rutas_list = []
        if not df_rutas.empty:
            df_api = df_rutas.rename(columns={
                'Zona Logística': 'zonaLogistica', 'Estación Origen (VACIAR)': 'estacionOrigen', 'Estación Destino (LLENAR)': 'estacionDestino',
                'Bicicletas a Mover': 'bicicletasAMover', 'Distancia (Km)': 'distanciaKm', 'Vehículo Asignado': 'vehiculoAsignado'
            })
            df_api['distanciaKm'] = df_api['distanciaKm'].replace({np.nan: None})
            rutas_list = df_api.to_dict(orient='records')
            
        payload_api = {
            "status": "success",
            "timestamp_evaluacion": str(ts_reciente),
            "metricas_globales": metricas,
            "hoja_de_ruta": rutas_list
        }

        print(json.dumps(payload_api, indent=4, ensure_ascii=False))
        print(f"\n[+] Ciclo predictivo completado. Próxima corrida en 60s...\n")

    except Exception as e:
        print(f"[-] Error en ejecución en vivo: {e}")
    
    time.sleep(60)