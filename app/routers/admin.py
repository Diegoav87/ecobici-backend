from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

from app.database import get_session
from app.dependencies import require_rol
from app.models.usuario import Usuario, Rol
from app.models.audit_log import AuditLog
from app.auth import hash_password

router = APIRouter(prefix="/admin", tags=["admin"])


class CrearUsuarioInput(BaseModel):
    nombre: str
    email: EmailStr
    password: str
    rol: Rol = Rol.viewer


class EditarUsuarioInput(BaseModel):
    nombre: Optional[str] = None
    rol: Optional[Rol] = None
    activo: Optional[bool] = None


class UsuarioResponse(BaseModel):
    id: int
    nombre: str
    email: str
    rol: Rol
    activo: bool
    created_at: datetime


class AuditLogResponse(BaseModel):
    id: int
    usuario_id: Optional[int]
    accion: str
    detalle: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime


@router.get("/usuarios", response_model=List[UsuarioResponse])
def listar_usuarios(
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin)),
):
    return session.exec(select(Usuario).order_by(Usuario.created_at.desc())).all()


@router.post("/usuarios", response_model=UsuarioResponse, status_code=201)
def crear_usuario(
    data: CrearUsuarioInput,
    request: Request,
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin)),
):
    existing = session.exec(select(Usuario).where(Usuario.email == data.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="El email ya está registrado")

    usuario = Usuario(
        nombre=data.nombre,
        email=data.email,
        password_hash=hash_password(data.password),
        rol=data.rol,
    )
    session.add(usuario)
    session.commit()
    session.refresh(usuario)

    log = AuditLog(
        usuario_id=current_user.id,
        accion="CREAR_USUARIO",
        detalle=f"email={data.email}, rol={data.rol}",
        ip_address=request.client.host,
    )
    session.add(log)
    session.commit()

    return usuario


@router.patch("/usuarios/{usuario_id}", response_model=UsuarioResponse)
def editar_usuario(
    usuario_id: int,
    data: EditarUsuarioInput,
    request: Request,
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin)),
):
    usuario = session.get(Usuario, usuario_id)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    cambios = []
    if data.nombre is not None:
        usuario.nombre = data.nombre
        cambios.append(f"nombre={data.nombre}")
    if data.rol is not None:
        usuario.rol = data.rol
        cambios.append(f"rol={data.rol}")
    if data.activo is not None:
        usuario.activo = data.activo
        cambios.append(f"activo={data.activo}")

    session.add(usuario)
    session.commit()
    session.refresh(usuario)

    log = AuditLog(
        usuario_id=current_user.id,
        accion="EDITAR_USUARIO",
        detalle=f"usuario_id={usuario_id}, cambios: {', '.join(cambios)}",
        ip_address=request.client.host,
    )
    session.add(log)
    session.commit()

    return usuario


@router.get("/audit-log", response_model=List[AuditLogResponse])
def audit_log(
    session: Session = Depends(get_session),
    current_user=Depends(require_rol(Rol.admin)),
):
    return session.exec(
        select(AuditLog).order_by(AuditLog.timestamp.desc())
    ).all()
