import json
from datetime import datetime
from sqlmodel import Session

from app.ml.prediccion_realtime import cargar_modelos, ejecutar_prediccion

from app.models.prediccion import Prediccion
from app.models.ruta import Ruta

# Los modelos se cargan una sola vez al importar este módulo
_modelos = None


def get_modelos():
    global _modelos
    if _modelos is None:
        _modelos = cargar_modelos()
    return _modelos


def correr_prediccion_y_guardar(session: Session) -> Prediccion:
    model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base = get_modelos()

    payload = ejecutar_prediccion(model_regresor, model_clasificador, mapa_historico, encoding_map, coords_base)

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
