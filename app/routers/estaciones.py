# app/routers/estaciones.py
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/estaciones", tags=["estaciones"])

URL_INFO   = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_information.json"
URL_STATUS = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_status.json"

@router.get("")
def get_estaciones():
    try:
        info   = requests.get(URL_INFO,   timeout=10).json()
        status = requests.get(URL_STATUS, timeout=10).json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo contactar la API de Ecobici: {e}")

    info_map = {s["station_id"]: s for s in info["data"]["stations"]}

    result = []
    for s in status["data"]["stations"]:
        sid = s["station_id"]
        if sid not in info_map:
            continue
        i        = info_map[sid]
        capacity = max(int(i["capacity"]), 1)
        bikes    = int(s["num_bikes_available"])
        result.append({
            "station_id":   sid,
            "name":         i["name"],
            "lat":          float(i["lat"]),
            "lon":          float(i["lon"]),
            "capacity":     capacity,
            "bikes_avail":  bikes,
            "docks_avail":  capacity - bikes,
            "pct_full":     round(bikes / capacity, 3),
            "is_renting":   int(s["is_renting"]),
            "is_returning": int(s["is_returning"]),
        })
    return result