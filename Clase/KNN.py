import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 1. Cargar los datos
df = pd.read_csv('entrenamiento2.csv', index_col='ID')  # Asumiendo que unes los 50 en un solo archivo
atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']

X_total = df[atributos]
# Escogemos la clase: ¿Es Guerrero o no?
y_total = (df['Clase'] == 'Guerrero').astype(int)

# 2. Dividir: 80% Entrenamiento y 20% Prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X_total, y_total, test_size=0.20, random_state=42
)

# 3. Iniciar y Entrenar ambos modelos (usando solo el 80% de entrenamiento)
modelo_knn = KNeighborsClassifier(n_neighbors=3)  # Agrupando por 3 vecinos cercanos
modelo_log = LogisticRegression()

modelo_knn.fit(X_entrenamiento, y_entrenamiento)
modelo_log.fit(X_entrenamiento, y_entrenamiento)


# 4. Crear una función para calcular: Eficiencia = Aciertos / Pruebas
def mostrar_eficiencia(modelo, X_datos, y_datos, nombre_fase):
    # El modelo hace sus predicciones
    predicciones = modelo.predict(X_datos)

    # Contamos cuántas veces la predicción es igual a la realidad
    aciertos = sum(predicciones == y_datos)
    pruebas = len(y_datos)

    # Calculamos la eficiencia
    eficiencia = aciertos / pruebas

    print(f"[{nombre_fase}] -> {aciertos} aciertos / {pruebas} pruebas = Eficiencia del {eficiencia * 100:.1f}%")


# 5. Ejecutar las pruebas solicitadas
print("resultados KNN ")
mostrar_eficiencia(modelo_knn, X_prueba, y_prueba, "20% Prueba")
mostrar_eficiencia(modelo_knn, X_total, y_total, "100% Total")

print("\n resultados regresion logistica")
mostrar_eficiencia(modelo_log, X_prueba, y_prueba, "20% Prueba")
mostrar_eficiencia(modelo_log, X_total, y_total, "100% Total")