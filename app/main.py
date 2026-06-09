from fastapi import FastAPI
from app.database import init_db
from app.routers import auth

app = FastAPI(title="Ecobici API")


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(auth.router)


@app.get("/")
def root():
    return {"status": "ok"}
