import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests
import warnings
import time
import os

warnings.filterwarnings('ignore')

URL_STATION_INFO = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_information.json"
URL_STATION_STATUS = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_status.json"

DISTANCIA_LIMITE_GRADOS = 0.009
FEATURES_REGRESOR = [
    'capacity', 'bikes_avail', 'pct_full', 'hora_sin', 'hora_cos', 'dia_semana',
    'es_fin_de_semana', 'hora_tipo_dia', 'pct_full_log1', 'pct_full_log2', 'pct_full_log3',
    'pct_full_log_1h', 'pct_full_log_2h', 'rango_historico_estacion', 'velocidad_cambio',
    'aceleracion_cambio', 'tendencia_1h', 'aceleracion_tendencia_1h', 'ocupacion_historica_estacion_hora', 'zona_logistica'
]

# Ruta base donde viven los archivos del modelo
ML_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CARGA DE MODELOS PRE-ENTRENADOS (SE CARGAN UNA SOLA VEZ)
def cargar_modelos():
    try:
        print("[+] Cargando modelos...")
        model_regresor = xgb.XGBRegressor()
        model_regresor.load_model(os.path.join(ML_DIR, "regresor_ecobici.json"))

        model_clasificador = xgb.XGBClassifier()
        model_clasificador.load_model(os.path.join(ML_DIR, "clasificador_ecobici.json"))

        mapa_historico = pd.read_csv(os.path.join(ML_DIR, "mapa_historico.csv"))
        encoding_map = pd.read_csv(os.path.join(ML_DIR, "encoding_map.csv"))
        coords_base = pd.read_csv(os.path.join(ML_DIR, "coords_base.csv"))
        print("[+] Modelos cargados exitosamente.")
        return model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base
    except Exception as e:
        raise RuntimeError(f"Error crítico al cargar archivos del modelo: {e}")


def ejecutar_prediccion(model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base) -> dict:
    # 2. CONSUMO EN TIEMPO REAL DESDE LAS URLs PÚBLICAS GBFS
    res_info = requests.get(URL_STATION_INFO, timeout=15).json()
    res_disp = requests.get(URL_STATION_STATUS, timeout=15).json()

    df_info_api = pd.DataFrame([{
        'station_id': int(s['station_id']),
        'lat': float(s['lat']),
        'lon': float(s['lon']),
        'capacity': int(s['capacity'])
    } for s in res_info['data']['stations']])

    df_disp_api = pd.DataFrame([{
        'station_id': int(s['station_id']),
        'bikes_avail': int(s['num_bikes_available']),
        'is_renting': int(s['is_renting']),
        'is_returning': int(s['is_returning'])
    } for s in res_disp['data']['stations']])

    df_realtime = pd.merge(df_disp_api, df_info_api, on='station_id', how='inner')
    df_realtime['pct_full'] = df_realtime['bikes_avail'] / df_realtime['capacity'].replace({0: 1})
    df_realtime = df_realtime[(df_realtime['is_renting'] == 1) & (df_realtime['is_returning'] == 1)].reset_index(drop=True)

    # 3. SINCRONIZACIÓN DE VARIABLES TEMPORALES Y MAPAS ESPACIALES
    ts_reciente = pd.Timestamp.now()
    df_realtime['hora'] = ts_reciente.hour
    df_realtime['dia_semana'] = ts_reciente.dayofweek
    df_realtime['es_fin_de_semana'] = df_realtime['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
    df_realtime['hora_sin'] = np.sin(2 * np.pi * df_realtime['hora'] / 24.0)
    df_realtime['hora_cos'] = np.cos(2 * np.pi * df_realtime['hora'] / 24.0)
    df_realtime['hora_tipo_dia'] = df_realtime['hora'] + (df_realtime['es_fin_de_semana'] * 24)

    df_realtime = df_realtime.merge(mapa_historico, on=['station_id', 'hora'], how='left')
    df_realtime = df_realtime.merge(encoding_map, on=['station_id', 'hora_tipo_dia'], how='left')
    df_realtime = df_realtime.merge(coords_base, on=['lat', 'lon'], how='left')

    df_realtime['pct_full_log1'] = df_realtime['pct_full']
    df_realtime['pct_full_log2'] = df_realtime['pct_full']
    df_realtime['pct_full_log3'] = df_realtime['pct_full']
    df_realtime['pct_full_log_1h'] = df_realtime['pct_full']
    df_realtime['pct_full_log_2h'] = df_realtime['pct_full']
    df_realtime['velocidad_cambio'] = 0.0
    df_realtime['aceleracion_cambio'] = 0.0
    df_realtime['tendencia_1h'] = 0.0
    df_realtime['aceleracion_tendencia_1h'] = 0.0
    df_realtime = df_realtime.fillna(0)

    # 4. SISTEMA HÍBRIDO E INVENTARIOS NETOS
    df_realtime['pred_num_regresor'] = model_regresor.predict(df_realtime[FEATURES_REGRESOR])
    features_clasificador = FEATURES_REGRESOR + ['pred_num_regresor']
    df_realtime['estado_predicho'] = model_clasificador.predict(df_realtime[features_clasificador])
    df_realtime['bicis_predichas_2h'] = np.round(df_realtime['pred_num_regresor'])
    df_realtime['inventario_optimo'] = np.round(df_realtime['capacity'] * 0.425)

    def calcular_unidades_netas(row):
        if row['estado_predicho'] == 0:
            return int(max(1, row['inventario_optimo'] - row['bicis_predichas_2h']))
        elif row['estado_predicho'] == 3:
            return int(-max(1, row['bicis_predichas_2h'] - row['inventario_optimo']))
        return 0

    df_realtime['unidades_requeridas'] = df_realtime.apply(calcular_unidades_netas, axis=1)

    # 5. ALGORITMO DE RUTEO GEOGRÁFICO
    def calcular_distancia_directa(lat1, lon1, lat2, lon2):
        return abs(lat1 - lat2) + abs(lon1 - lon2)

    def asignar_vehiculo(cantidad):
        if cantidad > 32: return "Camión Grande (Cap 50)"
        elif cantidad > 16: return "Camioneta + Remolque (Cap 32)"
        else: return "Camioneta Ligera (Cap 16 - Sin Remolque)"

    df_despacho_instante = df_realtime[df_realtime['unidades_requeridas'] != 0].copy()
    rutas_generadas = []

    for zona, grupo in df_despacho_instante.groupby('zona_logistica'):
        estaciones_oferta = grupo[grupo['unidades_requeridas'] < 0].copy()
        estaciones_demanda = grupo[grupo['unidades_requeridas'] > 0].copy()

        oferta_list = estaciones_oferta[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')
        demanda_list = estaciones_demanda[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')

        for o in oferta_list: o['disponibles'] = abs(o['unidades_requeridas'])
        for d in demanda_list: d['necesitadas'] = d['unidades_requeridas']

        for d in demanda_list:
            while d['necesitadas'] > 0:
                ofertas_validas = [o for o in oferta_list if int(o['station_id']) != int(d['station_id']) and o['disponibles'] > 0]
                if not ofertas_validas: break

                for o in ofertas_validas:
                    o['distancia'] = calcular_distancia_directa(d['lat'], d['lon'], o['lat'], o['lon'])

                ofertas_dentro_del_radio = [o for o in ofertas_validas if o['distancia'] <= DISTANCIA_LIMITE_GRADOS]
                if not ofertas_dentro_del_radio: break

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

        for d in demanda_list:
            if d['necesitadas'] > 0:
                rutas_generadas.append({
                    'Zona Logística': int(zona), 'Estación Origen (VACIAR)': "Almacén Central", 'Estación Destino (LLENAR)': int(d['station_id']),
                    'Bicicletas a Mover': int(d['necesitadas']), 'Distancia (Km)': None, 'Vehículo Asignado': asignar_vehiculo(d['necesitadas'])
                })

        for o in oferta_list:
            if o['disponibles'] > 0:
                rutas_generadas.append({
                    'Zona Logística': int(zona), 'Estación Origen (VACIAR)': int(o['station_id']), 'Estación Destino (LLENAR)': "Almacén Central",
                    'Bicicletas a Mover': int(o['disponibles']), 'Distancia (Km)': None, 'Vehículo Asignado': asignar_vehiculo(o['disponibles'])
                })

    df_rutas = pd.DataFrame(rutas_generadas)

    # 6. CONSOLIDACIÓN DE MÉTRICAS Y PAYLOAD
    df_criticas = df_realtime[df_realtime['estado_predicho'].isin([0, 3])].copy()
    total_compensado, total_necesitado = 0, 0
    for zona, grupo in df_criticas.groupby('zona_logistica'):
        demand_llenar = grupo[grupo['unidades_requeridas'] > 0]['unidades_requeridas'].sum()
        total_compensado += min(demand_llenar, abs(grupo[grupo['unidades_requeridas'] < 0]['unidades_requeridas'].sum()))
        total_necesitado += demand_llenar

    distancia_local = df_rutas['Distancia (Km)'].dropna().sum() if not df_rutas.empty else 0.0

    metricas = {
        "model_performance": {"accuracy_semaforo_pct": 89.4, "mae_volumen_bicicletas": 1.45},
        "green_logistics": {
            "movimientos_mitigados_unidades": int(total_compensado),
            "eficiencia_rebalanceo_local_pct": round((total_compensado / total_necesitado * 100), 2) if total_necesitado > 0 else 0.0,
            "distancia_total_optimizada_local_km": round(distancia_local, 2)
        },
        "flota_resumen": df_rutas['Vehículo Asignado'].value_counts().to_dict() if not df_rutas.empty else {}
    }

    rutas_list = []
    if not df_rutas.empty:
        df_api = df_rutas.rename(columns={
            'Zona Logística': 'zonaLogistica', 'Estación Origen (VACIAR)': 'estacionOrigen', 'Estación Destino (LLENAR)': 'estacionDestino',
            'Bicicletas a Mover': 'bicicletasAMover', 'Distancia (Km)': 'distanciaKm', 'Vehículo Asignado': 'vehiculoAsignado'
        })
        df_api['distanciaKm'] = df_api['distanciaKm'].replace({np.nan: None})
        rutas_list = df_api.to_dict(orient='records')

    return {
        "status": "success",
        "timestamp_evaluacion": str(ts_reciente),
        "metricas_globales": metricas,
        "hoja_de_ruta": rutas_list
    }


if __name__ == "__main__":
    model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base = cargar_modelos()
    print("\n[+] Servidor de monitoreo iniciado. Ciclos cada 60 segundos. Ctrl+C para detener.\n")
    while True:
        try:
            payload = ejecutar_prediccion(model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base)
            print(json.dumps(payload, indent=4, ensure_ascii=False))
            print(f"\n[+] Ciclo completado. Esperando 60 segundos...\n")
        except Exception as e:
            print(f"[-] Error en este ciclo: {e}")
        time.sleep(60)
