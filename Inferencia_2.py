import joblib

# Datos fijos para prueba
features = [[2000, 500000, 5]]
# Cargar el modelo
modelo = joblib.load("modelo_regresion1.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
