from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd  # lo agrego para crear el DataFrame

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("modelo_regresion1.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, year: float = Form(...), km_driven: float = Form(...), seats: float = Form(...)):
    # Crear un DataFrame con las características para la predicción
    features = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'seats': [seats]
    })
    prediction = model.predict(features)[0]
    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "result": f"Predicción: Clase {prediction}"
        }
    )


