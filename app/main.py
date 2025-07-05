from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import json

app = FastAPI()

class ProyectoInput(BaseModel):
    proyecto_id: int

@app.get("/")
def raiz():
    return {"mensaje": "Hola Mundo para API"}

# -----------------------------
# 1. PREDECIR CON MODELO
# -----------------------------
@app.post("/predecir/")
def predecir_sinceridad(input: ProyectoInput):
    try:
        modelo = joblib.load(os.path.join("app", "modelo", "Modelo26Junio.pkl"))
        dataset = pd.read_csv(os.path.join("app", "data", "DataSetParaEntrenamiento25Junio.csv"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo o dataset: {e}")

    proyecto = dataset[dataset["proyecto_idproyecto"] == input.proyecto_id].sort_values(by="id_visita")
    if proyecto.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el proyecto {input.proyecto_id}")

    # Crear variables
    proyecto['sinceridad_anterior'] = proyecto['SinceridadAcumuladaTopadaMetodo3'].shift(1)
    proyecto['diferencia_anterior'] = proyecto['diferencia'].shift(1)
    proyecto['porcentaje_real_anterior'] = proyecto['porcentaje_real'].shift(1)
    proyecto['promedio_sinceridad_antes'] = proyecto['SinceridadAcumuladaTopadaMetodo3'].expanding().mean().shift()
    proyecto = proyecto.dropna()

    columnas = [
        'porcentaje_real', 'diferencia', 'proyecto_monto', 'proyecto_duracion',
        'id_visita', 'm2c', 'sinceridad_anterior', 'diferencia_anterior',
        'porcentaje_real_anterior', 'promedio_sinceridad_antes'
    ]

    X = proyecto[columnas]
    y_pred = modelo.predict(X)
    proyecto['Sinceridad Predecida'] = y_pred

    # Crear gráfica
    df_plot = proyecto.groupby('id_visita')[['Sinceridad Predecida']].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot['id_visita'], df_plot['Sinceridad Predecida'],
             marker='o', linestyle='--', label='Índice de Sinceridad Predecido')
    plt.xlabel("Visita")
    plt.ylabel("Sinceridad")
    plt.title(f"Sinceridad Predecida - Proyecto {input.proyecto_id}")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("app/temp", exist_ok=True)
    img_path = f"app/temp/grafica_{input.proyecto_id}.png"
    plt.savefig(img_path)
    plt.close()

    # Crear JSON
    resultados = [
        {
            "id_visita": int(row["id_visita"]),
            "sinceridad_predicha": float(row["Sinceridad Predecida"])
        } for _, row in proyecto.iterrows()
    ]

    json_path = f"app/temp/predicciones_{input.proyecto_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "proyecto_id": input.proyecto_id,
            "predicciones": resultados
        }, f, indent=4, ensure_ascii=False)

    return {
        "proyecto_id": input.proyecto_id,
        "grafica_url": f"/grafica/{input.proyecto_id}",
        "json_url": f"/json/{input.proyecto_id}",
        "predicciones": resultados
    }

# -----------------------------
# 2. CARGAR JSON YA EXISTENTE
# -----------------------------
@app.get("/cargarjson/{proyecto_id}")
def cargar_json_existente(proyecto_id: int):
    json_path = os.path.join("app", "data", f"predicciones_{proyecto_id}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Archivo JSON no encontrado en data/")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            datos_json = json.load(f)

        predicciones = datos_json.get("predicciones")
        if not predicciones:
            raise HTTPException(status_code=400, detail="No hay predicciones en el archivo JSON")

        df = pd.DataFrame(predicciones)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo JSON: {e}")

    # Graficar
    df_plot = df.groupby("id_visita")[["sinceridad_predicha"]].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["id_visita"], df_plot["sinceridad_predicha"],
             marker='o', linestyle='--', label='Índice de Sinceridad Promedio')
    plt.xlabel("Visita")
    plt.ylabel("Sinceridad")
    plt.title(f"Sinceridad Promedio por Visita - Proyecto {proyecto_id}")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("app/temp", exist_ok=True)
    grafica_path = f"app/temp/grafica_{proyecto_id}.png"
    plt.savefig(grafica_path)
    plt.close()

    # Guardar JSON con promedio
    json_resumen = {
        "proyecto_id": proyecto_id,
        "promedio_por_visita": [
            {"id_visita": int(row["id_visita"]), "promedio_sinceridad": float(row["sinceridad_predicha"])}
            for _, row in df_plot.iterrows()
        ]
    }

    resumen_path = f"app/temp/predicciones_promedio_{proyecto_id}.json"
    with open(resumen_path, "w", encoding="utf-8") as f:
        json.dump(json_resumen, f, indent=4, ensure_ascii=False)

    return {
        "proyecto_id": proyecto_id,
        "grafica_url": f"/grafica/{proyecto_id}",
        "json_url": f"/json/promedio/{proyecto_id}",
        "promedio_por_visita": json_resumen["promedio_por_visita"]
    }

# -----------------------------
# 3. SERVICIOS PARA DESCARGAS
# -----------------------------
@app.get("/grafica/{proyecto_id}")
def obtener_grafica(proyecto_id: int):
    img_path = f"app/temp/grafica_{proyecto_id}.png"
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/png", filename=f"grafica_{proyecto_id}.png")
    else:
        raise HTTPException(status_code=404, detail="Gráfica no encontrada")

@app.get("/json/{proyecto_id}")
def obtener_json(proyecto_id: int):
    json_path = f"app/temp/predicciones_{proyecto_id}.json"
    if os.path.exists(json_path):
        return FileResponse(json_path, media_type="application/json", filename=f"predicciones_{proyecto_id}.json")
    else:
        raise HTTPException(status_code=404, detail="Archivo JSON no encontrado")

@app.get("/json/promedio/{proyecto_id}")
def obtener_json_resumen(proyecto_id: int):
    resumen_path = f"app/temp/predicciones_promedio_{proyecto_id}.json"
    if os.path.exists(resumen_path):
        return FileResponse(resumen_path, media_type="application/json", filename=f"predicciones_promedio_{proyecto_id}.json")
    else:
        raise HTTPException(status_code=404, detail="Archivo JSON promedio no encontrado")
