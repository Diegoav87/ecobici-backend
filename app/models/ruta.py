from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Ruta(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prediccion_id: int = Field(foreign_key="prediccion.id")
    zona_logistica: int
    estacion_origen: str
    estacion_destino: str
    bicicletas_a_mover: int
    distancia_km: Optional[float] = None
    vehiculo_asignado: str
    completada: bool = Field(default=False)
    completada_por: Optional[int] = Field(default=None, foreign_key="usuario.id")
    completada_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
