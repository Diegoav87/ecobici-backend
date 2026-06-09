from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session
from datetime import datetime

from app.database import get_session
from app.dependencies import require_rol
from app.models.usuario import Rol
from app.models.ruta import Ruta
from app.models.audit_log import AuditLog

router = APIRouter(prefix="/rutas", tags=["rutas"])


@router.patch("/{ruta_id}/completar")
def completar_ruta(
    ruta_id: int,
    request: Request,
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin, Rol.operador)),
):
    ruta = session.get(Ruta, ruta_id)
    if not ruta:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    if ruta.completada:
        raise HTTPException(status_code=400, detail="La ruta ya fue marcada como completada")

    ruta.completada = True
    ruta.completada_por = current_user.id
    ruta.completada_at = datetime.utcnow()
    session.add(ruta)

    log = AuditLog(
        usuario_id=current_user.id,
        accion="COMPLETAR_RUTA",
        detalle=f"ruta_id={ruta_id}, origen={ruta.estacion_origen}, destino={ruta.estacion_destino}",
        ip_address=request.client.host,
    )
    session.add(log)
    session.commit()
    session.refresh(ruta)

    return ruta
