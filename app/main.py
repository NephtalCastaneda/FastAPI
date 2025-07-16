from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

class ProyectoInput(BaseModel):
    proyecto_id: int

@app.get("/")
def raiz():
    return {"mensaje": "API en FastAPI funcionando correctamente"}

@app.post("/conexion-api")
def conexion_api(input: ProyectoInput):
    try:
        # Cargar modelo y dataset
        modelo = joblib.load("modelo/Modelo26Junio.pkl")
        df = pd.read_csv("data/DataSetParaEntrenamiento25Junio.csv")
        df = df[df["proyecto_idproyecto"] == input.proyecto_id].sort_values(by="id_visita")

        if df.empty:
            raise HTTPException(status_code=404, detail="Proyecto no encontrado")

        # Preparar columnas derivadas
        df['sinceridad_anterior'] = df['SinceridadAcumuladaTopadaMetodo3'].shift(1)
        df['diferencia_anterior'] = df['diferencia'].shift(1)
        df['porcentaje_real_anterior'] = df['porcentaje_real'].shift(1)
        df['promedio_sinceridad_antes'] = df['SinceridadAcumuladaTopadaMetodo3'].expanding().mean().shift()
        df = df.dropna()

        columnas = [
            'porcentaje_real', 'diferencia', 'proyecto_monto', 'proyecto_duracion',
            'id_visita', 'm2c', 'sinceridad_anterior', 'diferencia_anterior',
            'porcentaje_real_anterior', 'promedio_sinceridad_antes'
        ]

        for col in columnas:
            if col not in df.columns:
                raise HTTPException(status_code=422, detail=f"Falta la columna requerida: {col}")

        X = df[columnas]
        df['Sinceridad Predecida'] = modelo.predict(X)

        resultado = df[['id_visita', 'porcentaje_real', 'porcentaje_programado', 'diferencia', 'Sinceridad Predecida']].to_dict(orient='records')

        return {
            "proyecto_id": input.proyecto_id,
            "indice_sinceridad": float(df["Sinceridad Predecida"].mean()),
            "datos": [
                {
                    "visita": int(row["id_visita"]),
                    "real": float(row["porcentaje_real"]),
                    "programado": float(row["porcentaje_programado"]),
                    "diferencia": float(row["diferencia"]),
                    "indice": float(row["Sinceridad Predecida"])
                }
                for row in resultado
            ]
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
