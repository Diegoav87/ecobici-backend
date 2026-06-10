from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Prediccion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp_evaluacion: datetime
    movimientos_mitigados_unidades: int
    eficiencia_rebalanceo_local_pct: float
    distancia_total_optimizada_local_km: float
    flota_resumen: str  # JSON serializado
    created_at: datetime = Field(default_factory=datetime.utcnow)
