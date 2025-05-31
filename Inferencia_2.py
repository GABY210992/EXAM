import joblib

# Datos fijos para prueba
features = [[2000, 500000, 20,5, 1500, 100, 4,'Diesel','Individual','Manual','Second Owner' ]]
# Cargar el modelo
modelo = joblib.load("modelo_regresion.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
