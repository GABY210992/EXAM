from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Carga tu modelo
modelo_regresion = joblib.load("modelo_regresion1.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    # Mostrar formulario vacío
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    year: int = Form(...),
    km_driven: int = Form(...),
    seats: int = Form(...)
):
    # Crear DataFrame con los datos recibidos
    nuevo_caso = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'seats': [seats]
    })

    # Predecir usando el modelo cargado
    prediccion = modelo_regresion.predict(nuevo_caso)[0]

    # Mostrar resultado en el formulario
    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "result": f"El valor estimado es: {prediccion:.2f}"
        }
    )

