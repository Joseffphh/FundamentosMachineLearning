import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# paso 1 es leer los datos del archivo generados.
df = pd.read_csv("datos_ventas.csv", names=["X", "Y", "clase"]) #names asigna nombres a las columnas, x, y, clase en este caso.

#paso 2 es separar las caracteristicas (en una variable llamada caracteristicas) X y Y y la clase(en una variable clase).
caracteristicas = df[["X","Y"]]
etiquetas = df["clase"]

#paso 3 es crear el modelo KNN(usando la función de la libreria scikit)
modelo_knn = KNeighborsClassifier(n_neighbors=10) #KNeighborsClassifier es el criterio que ponemos
#n__neighbors le decimos que se fije en los 10 vecinos mas cercanos.

#paso 4 es entrenar el modelo con nuestros datos.
modelo_knn.fit(caracteristicas, etiquetas)
#.fit lo que hace es el ajuste, en la funcion de "entrenar", le entregamos los datos, que es la variable caracteristica
#y las respuestas que es la variable "etiquetas", esta función no hace calculos.

#paso 5 hacer la predicción de datos con nuevos datos
#en este caso, inventamos un punto en X=-5, Y=5, que nos deberia dar clase 0.
nuevo_punto = pd.DataFrame([{"X": -5, "Y":5}])

#variable predicción para imprimir los resultados.
prediccion = modelo_knn.predict(nuevo_punto)
#es el que hace toda la predicción de los datos, le pasamos las coordenadas
#y memoriza  lo que hizo .fit, mide la distancia entre el "nuevo_punto" y todos los demas
#puntos del archivo y encuentra cuales son los 10 mas cercanos
#hace la votación entre esos 10 y devuelve la clase ganadora

print(prediccion)