import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from sqlmodel import Session
from app.database import init_db, engine
from app.routers import auth, predicciones, rutas, admin, estaciones
from app.services.modelo import correr_prediccion_y_guardar

app = FastAPI(title="Ecobici API")

FRONTEND_URLS = [u.strip() for u in os.getenv("FRONTEND_URL", "http://localhost:5173").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INTERVALO_MINUTOS = 60


def tarea_prediccion_automatica():
    try:
        with Session(engine) as session:
            correr_prediccion_y_guardar(session)
        print("[+] Predicción automática completada.")
    except Exception as e:
        print(f"[-] Error en predicción automática: {e}")


@app.on_event("startup")
def on_startup():
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        tarea_prediccion_automatica, "interval", minutes=INTERVALO_MINUTOS
    )
    scheduler.start()
    print(f"[+] Scheduler iniciado. Predicciones cada {INTERVALO_MINUTOS} minutos.")


app.include_router(auth.router)
app.include_router(predicciones.router)
app.include_router(rutas.router)
app.include_router(admin.router)
app.include_router(estaciones.router)


@app.get("/")
def root():
    return {"status": "ok"}
