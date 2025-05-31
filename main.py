from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar modelo previamente entrenado
modelo_regresion = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    year: int = Form(...),
    km_driven: int = Form(...),
    seats: int = Form(...)
):
    nuevo_caso = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'seats': [seats]
    })

    prediccion = modelo_regresion.predict(nuevo_caso)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": f"{prediccion:.2f}"
