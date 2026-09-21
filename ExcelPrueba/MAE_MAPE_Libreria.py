import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

#leemos el documento de excel
df = pd.read_excel("Tres Tristes Tigres 1.xlsx", sheet_name='Hoja1')

#extraemos los datos reales.
y_real = df['Puntuación'][0:6].reset_index(drop=True)

#las predicciones están en la columna "Unnamed: 8", filas 30 a 35
y_pred = df['Unnamed: 8'][30:36].reset_index(drop=True)


print("version normal sin usar librerias: ")
#pandas permite hacer operaciones matemáticas directamente entre columnas.
#calculamos los errores absolutos y los promediamos (.mean())
mae_normal = abs(y_real - y_pred).mean()

#calculamos la diferencia porcentual absoluta y la promediamos
mape_normal = abs((y_real - y_pred) / y_real).mean() * 100

print(f"MAE calculado a mano: {mae_normal:.4f}")
print(f"MAPE calculado a mano: {mape_normal:.4f}%\n")




print("version con libreria scikitlearn")
#usamos las funciones de la librería en una sola línea de código

mae_libreria = mean_absolute_error(y_real, y_pred)
mape_libreria = mean_absolute_percentage_error(y_real, y_pred) * 100

print(f"MAE (librería): {mae_libreria:.4f}")
print(f"MAPE (librería): {mape_libreria:.4f}%")