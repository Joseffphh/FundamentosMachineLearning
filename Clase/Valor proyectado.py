import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Cargar datos de entrenamiento y prueba
df_train = pd.read_csv('entrenamiento.csv', index_col='ID')
df_test = pd.read_csv('prueba.csv', index_col='ID')
atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']

X_train = df_train[atributos]
y_train = df_train['Clase']
X_test = df_test[atributos]

# 2. Entrenar un solo modelo Multinomial
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Extraer las variables matemáticas de la regresión
b0 = modelo.intercept_  # Los valores de la columna b0
b = modelo.coef_  # Los valores b1, b2, b3, b4, b5
clases = modelo.classes_

# Elegimos el primer personaje del archivo de prueba (ID 31)
caso_nuevo = X_test.iloc[0].values
print(f"Atributos del Caso Nuevo: {caso_nuevo}\n")

# Paso 1: Calcular Z para cada clase
print("1) Cálculo de Z para cada clase:")
Z_valores = []

for i, clase in enumerate(clases):
    # Fórmula: Z = b0 + b1*atr1 + b2*atr2 + b3*atr3 + b4*atr4 + b5*atr5
    z_clase = b0[i] + np.dot(b[i], caso_nuevo)
    Z_valores.append(z_clase)
    print(f"Z {clase}: {z_clase:.4f}")

# Paso 2: Calcular la probabilidad usando la fórmula
print("\n2) Cálculo de la Probabilidad de cada clase:")

# Denominador de la fórmula: Sumatoria de e elevado a todos los valores Z
suma_euler_z = np.sum(np.exp(Z_valores))

for i, clase in enumerate(clases):
    # Numerador: e elevado a la Z de la clase actual
    euler_z_clase = np.exp(Z_valores[i])

    # Fórmula: e^Z / Sumatoria(e^Z)
    probabilidad = euler_z_clase / suma_euler_z

    print(f"P({clase}): {probabilidad * 100:.2f}%")