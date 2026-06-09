from fastapi import FastAPI
from app.database import init_db
from app.routers import auth, predicciones, rutas, admin

app = FastAPI(title="Ecobici API")


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(auth.router)
app.include_router(predicciones.router)
app.include_router(rutas.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {"status": "ok"}
