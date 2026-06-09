from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.routers import auth, predicciones, rutas, admin, estaciones

app = FastAPI(title="Ecobici API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(auth.router)
app.include_router(predicciones.router)
app.include_router(rutas.router)
app.include_router(admin.router)
app.include_router(estaciones.router)

@app.get("/")
def root():
    return {"status": "ok"}