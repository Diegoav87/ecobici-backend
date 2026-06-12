# Ecobici API — Backend

API REST desarrollada con **FastAPI** que sirve como núcleo del sistema de predicción y balanceo de la red de bicicletas Ecobici. Se encarga de la autenticación de usuarios, la orquestación del modelo de ML, la generación de rutas de redistribución y la exposición de datos en tiempo real de las estaciones.

---

## Tabla de contenidos

- [Arquitectura general](#arquitectura-general)
- [Stack tecnológico](#stack-tecnológico)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Variables de entorno](#variables-de-entorno)
- [Ejecutar el servidor](#ejecutar-el-servidor)
- [Documentación de la API](#documentación-de-la-api)
- [Endpoints](#endpoints)
- [Roles y permisos](#roles-y-permisos)
- [Tareas programadas](#tareas-programadas)
- [Integración con el modelo de ML](#integración-con-el-modelo-de-ml)
- [Estructura del proyecto](#estructura-del-proyecto)

## Stack tecnológico

| Componente              | Tecnología                         |
| ----------------------- | ---------------------------------- |
| Framework web           | FastAPI 0.136                      |
| Servidor ASGI           | Uvicorn                            |
| ORM                     | SQLModel (SQLAlchemy + Pydantic)   |
| Base de datos           | PostgreSQL (driver psycopg v3)     |
| Autenticación           | JWT (python-jose) + bcrypt/passlib |
| Tareas en segundo plano | APScheduler                        |
| Cliente HTTP            | requests                           |
| Validación de datos     | Pydantic v2                        |
| Procesamiento de datos  | pandas, numpy                      |

---

## Requisitos previos

- Python 3.10 o superior
- PostgreSQL 14 o superior corriendo en `localhost:5432`
- El servicio del modelo de ML corriendo en `http://localhost:8001` (ver `modelo-ml/`)

---

## Instalación

```bash
# 1. Ir al directorio del backend
cd backend

# 2. Crear y activar un entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -e .
```

> Si tienes `uv` instalado puedes usar `uv pip install -e .` para una instalación más rápida.

---

## Variables de entorno

Copia `.env.example` a `.env` y ajusta los valores:

```bash
cp .env.example .env
```

| Variable         | Descripción                                                | Ejemplo                                                 |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `DATABASE_URL`   | Cadena de conexión a PostgreSQL                            | `postgresql+psycopg://user:pass@localhost:5432/ecobici` |
| `SECRET_KEY`     | Clave secreta para firmar JWT (debe ser larga y aleatoria) | `cambia-esto-por-una-clave-larga`                       |
| `FRONTEND_URL`   | Origen(es) permitido(s) por CORS (separados por coma)      | `http://localhost:5173`                                 |
| `ML_SERVICE_URL` | URL base del servicio del modelo de ML                     | `http://localhost:8001`                                 |

---

## Ejecutar el servidor

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Al arrancar, el servidor:

1. Inicializa las tablas de la base de datos si no existen.
2. Lanza el scheduler que ejecuta una predicción automática cada 60 minutos.

Verifica que esté corriendo:

```bash
curl http://localhost:8000/
# { "status": "ok" }
```

---

## Documentación de la API

FastAPI genera documentación interactiva de forma automática:

| Interfaz   | URL                          |
| ---------- | ---------------------------- |
| Swagger UI | `http://localhost:8000/docs` |

---

## Endpoints

### Autenticación — `/auth`

| Método | Ruta             | Descripción                                        | Acceso      |
| ------ | ---------------- | -------------------------------------------------- | ----------- |
| `POST` | `/auth/register` | Registrar nuevo usuario (rol `viewer` por defecto) | Público     |
| `POST` | `/auth/login`    | Iniciar sesión, retorna JWT                        | Público     |
| `GET`  | `/auth/me`       | Obtener perfil del usuario autenticado             | Autenticado |

### Predicciones — `/predicciones`

| Método | Ruta                     | Descripción                                      | Acceso          |
| ------ | ------------------------ | ------------------------------------------------ | --------------- |
| `POST` | `/predicciones/ejecutar` | Ejecutar una predicción manualmente              | admin, operador |
| `GET`  | `/predicciones/latest`   | Obtener la predicción más reciente con sus rutas | Autenticado     |
| `GET`  | `/predicciones`          | Listar las últimas 20 predicciones (resumen)     | Autenticado     |

### Rutas de redistribución — `/rutas`

| Método  | Ruta                         | Descripción                     | Acceso          |
| ------- | ---------------------------- | ------------------------------- | --------------- |
| `PATCH` | `/rutas/{ruta_id}/completar` | Marcar una ruta como completada | admin, operador |

### Administración — `/admin`

| Método  | Ruta                           | Descripción                  | Acceso |
| ------- | ------------------------------ | ---------------------------- | ------ |
| `GET`   | `/admin/usuarios`              | Listar todos los usuarios    | admin  |
| `POST`  | `/admin/usuarios`              | Crear un usuario             | admin  |
| `PATCH` | `/admin/usuarios/{usuario_id}` | Editar un usuario            | admin  |
| `GET`   | `/admin/audit-log`             | Ver el registro de auditoría | admin  |

### Estaciones — `/estaciones`

| Método | Ruta          | Descripción                                           | Acceso      |
| ------ | ------------- | ----------------------------------------------------- | ----------- |
| `GET`  | `/estaciones` | Obtener estado en tiempo real de todas las estaciones | Autenticado |

---

## Roles y permisos

| Rol        | Descripción                                                        |
| ---------- | ------------------------------------------------------------------ |
| `admin`    | Acceso total: gestión de usuarios, auditoría, predicciones y rutas |
| `operador` | Puede ejecutar predicciones y completar rutas                      |
| `viewer`   | Solo lectura: consulta predicciones y estaciones                   |

---

## Tareas programadas

El backend usa **APScheduler** para ejecutar predicciones de forma automática. El intervalo está definido en `app/main.py`:

```python
INTERVALO_MINUTOS = 5
```

Cada ejecución:

1. Llama a `POST {ML_SERVICE_URL}/predecir`.
2. Procesa la respuesta (métricas globales + hoja de ruta).
3. Persiste los resultados en la base de datos (`prediccion` + `ruta`).
4. Registra la acción en el audit log.

---

## Integración con el modelo de ML

El servicio de ML (`modelo-ml/`) expone un único endpoint `POST /predecir`. El backend lo llama mediante `app/services/modelo.py` y espera la siguiente estructura de respuesta:

```json
{
  "timestamp_evaluacion": "2024-01-01T12:00:00",
  "metricas_globales": {
    "mae": 1.23,
    "rmse": 2.45,
    "flota_total": 150,
    "estaciones_evaluadas": 480
  },
  "hoja_de_ruta": [
    {
      "vehiculo": "Camioneta Ligera",
      "paradas": [
        {
          "estacion_id": "1",
          "nombre": "Reforma - Juárez",
          "accion": "recoger",
          "cantidad": 5
        }
      ]
    }
  ]
}
```

---

## Estructura del proyecto

```
backend/
├── app/
│   ├── main.py               # Punto de entrada, middleware CORS, scheduler
│   ├── auth.py               # Lógica de JWT y hashing de contraseñas
│   ├── database.py           # Configuración de SQLModel y engine
│   ├── dependencies.py       # Dependencias de inyección (auth, roles)
│   ├── models/
│   │   ├── usuario.py        # Modelo de usuario y roles
│   │   ├── prediccion.py     # Modelo de predicción del ML
│   │   ├── ruta.py           # Modelo de ruta de redistribución
│   │   └── audit_log.py      # Modelo de registro de auditoría
│   ├── routers/
│   │   ├── auth.py           # Endpoints de autenticación
│   │   ├── predicciones.py   # Endpoints de predicciones
│   │   ├── rutas.py          # Endpoints de rutas
│   │   ├── admin.py          # Endpoints de administración
│   │   └── estaciones.py     # Endpoints de estaciones en tiempo real
│   ├── services/
│   │   └── modelo.py         # Orquestación de llamadas al servicio ML
│   └── lib/                  # Utilidades varias
├── .env.example              # Plantilla de variables de entorno
├── .python-version           # Versión de Python requerida
└── pyproject.toml            # Metadatos y dependencias del proyecto
```
