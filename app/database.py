from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv
import os
import app.models  # noqa: F401 — registra los modelos antes de crear tablas

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
