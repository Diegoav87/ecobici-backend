from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json

from app.database import get_session
from app.dependencies import get_current_user, require_rol
from app.models.usuario import Rol
from app.models.prediccion import Prediccion
from app.models.ruta import Ruta
from app.models.audit_log import AuditLog
from app.services.modelo import correr_prediccion_y_guardar

router = APIRouter(prefix="/predicciones", tags=["predicciones"])


class RutaResponse(BaseModel):
    id: int
    zona_logistica: int
    estacion_origen: str
    estacion_destino: str
    bicicletas_a_mover: int
    distancia_km: Optional[float]
    vehiculo_asignado: str
    completada: bool
    completada_at: Optional[datetime]


class PrediccionResponse(BaseModel):
    id: int
    timestamp_evaluacion: datetime
    movimientos_mitigados_unidades: int
    eficiencia_rebalanceo_local_pct: float
    distancia_total_optimizada_local_km: float
    flota_resumen: dict
    created_at: datetime
    rutas: List[RutaResponse] = []


class PrediccionResumenResponse(BaseModel):
    id: int
    timestamp_evaluacion: datetime
    movimientos_mitigados_unidades: int
    eficiencia_rebalanceo_local_pct: float
    distancia_total_optimizada_local_km: float
    flota_resumen: dict
    created_at: datetime


@router.post("/ejecutar", response_model=PrediccionResponse)
def ejecutar(
    request: Request,
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin, Rol.operador)),
):
    try:
        prediccion = correr_prediccion_y_guardar(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar el modelo: {e}")

    log = AuditLog(
        usuario_id=current_user.id,
        accion="EJECUTAR_PREDICCION",
        detalle=f"prediccion_id={prediccion.id}",
        ip_address=request.client.host,
    )
    session.add(log)
    session.commit()

    return _build_response(prediccion, session)


@router.get("/latest", response_model=PrediccionResponse)
def latest(
    session: Session = Depends(get_session),
    current_user=Depends(get_current_user),
):
    prediccion = session.exec(
        select(Prediccion).order_by(Prediccion.created_at.desc())
    ).first()

    if not prediccion:
        raise HTTPException(status_code=404, detail="No hay predicciones registradas")

    return _build_response(prediccion, session)


@router.get("", response_model=List[PrediccionResumenResponse])
def listar(
    session: Session = Depends(get_session),
    current_user=Depends(get_current_user),
):
    predicciones = session.exec(
        select(Prediccion).order_by(Prediccion.created_at.desc()).limit(20)
    ).all()

    return [
        {**p.model_dump(), "flota_resumen": json.loads(p.flota_resumen)}
        for p in predicciones
    ]


def _build_response(prediccion: Prediccion, session: Session) -> dict:
    rutas = session.exec(
        select(Ruta).where(Ruta.prediccion_id == prediccion.id)
    ).all()

    return {
        **prediccion.model_dump(),
        "flota_resumen": json.loads(prediccion.flota_resumen),
        "rutas": rutas,
    }
