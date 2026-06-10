from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from pydantic import BaseModel, EmailStr
from app.database import get_session
from app.models.usuario import Usuario, Rol
from app.models.audit_log import AuditLog
from app.auth import hash_password, verify_password, create_access_token
from app.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterInput(BaseModel):
    nombre: str
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UsuarioResponse(BaseModel):
    id: int
    nombre: str
    email: str
    rol: Rol
    activo: bool


@router.post("/register", response_model=UsuarioResponse, status_code=status.HTTP_201_CREATED)
def register(data: RegisterInput, session: Session = Depends(get_session)):
    existing = session.exec(select(Usuario).where(Usuario.email == data.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="El email ya está registrado")

    usuario = Usuario(
        nombre=data.nombre,
        email=data.email,
        password_hash=hash_password(data.password),
        rol= Rol.viewer 
    )
    session.add(usuario)
    session.commit()
    session.refresh(usuario)
    return usuario


@router.post("/login", response_model=TokenResponse)
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    usuario = session.exec(select(Usuario).where(Usuario.email == form_data.username)).first()

    if not usuario or not verify_password(form_data.password, usuario.password_hash):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    if not usuario.activo:
        raise HTTPException(status_code=403, detail="Usuario desactivado")

    token = create_access_token({"sub": str(usuario.id), "rol": usuario.rol})

    log = AuditLog(
        usuario_id=usuario.id,
        accion="LOGIN",
        ip_address=request.client.host,
    )
    session.add(log)
    session.commit()

    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UsuarioResponse)
def me(current_user: Usuario = Depends(get_current_user)):
    return current_user
