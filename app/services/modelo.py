import json
import os
import requests
from datetime import datetime
from sqlmodel import Session
from dotenv import load_dotenv

from app.models.prediccion import Prediccion
from app.models.ruta import Ruta

load_dotenv()

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")


def correr_prediccion_y_guardar(session: Session) -> Prediccion:
    try:
        response = requests.post(f"{ML_SERVICE_URL}/predecir", timeout=60)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"No se pudo contactar el servicio ML: {e}")

    metricas = payload["metricas_globales"]
    perf = metricas["model_performance"]
    logistics = metricas["green_logistics"]

    prediccion = Prediccion(
        timestamp_evaluacion=datetime.fromisoformat(payload["timestamp_evaluacion"]),
        accuracy_semaforo_pct=perf["accuracy_semaforo_pct"],
        mae_volumen_bicicletas=perf["mae_volumen_bicicletas"],
        movimientos_mitigados_unidades=logistics["movimientos_mitigados_unidades"],
        eficiencia_rebalanceo_local_pct=logistics["eficiencia_rebalanceo_local_pct"],
        distancia_total_optimizada_local_km=logistics["distancia_total_optimizada_local_km"],
        flota_resumen=json.dumps(metricas["flota_resumen"], ensure_ascii=False),
    )
    session.add(prediccion)
    session.commit()
    session.refresh(prediccion)

    for r in payload["hoja_de_ruta"]:
        ruta = Ruta(
            prediccion_id=prediccion.id,
            zona_logistica=r["zonaLogistica"],
            estacion_origen=str(r["estacionOrigen"]),
            estacion_destino=str(r["estacionDestino"]),
            bicicletas_a_mover=r["bicicletasAMover"],
            distancia_km=r.get("distanciaKm"),
            vehiculo_asignado=r["vehiculoAsignado"],
        )
        session.add(ruta)

    session.commit()
    return prediccion
