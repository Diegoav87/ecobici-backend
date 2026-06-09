from sqlmodel import SQLModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


class Rol(str, Enum):
    admin = "admin"
    operador = "operador"
    viewer = "viewer"


class Usuario(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    nombre: str
    email: str = Field(unique=True, index=True)
    password_hash: str
    rol: Rol = Field(default=Rol.viewer)
    activo: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
