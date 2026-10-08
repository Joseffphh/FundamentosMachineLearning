import pandas as pd
import numpy as np

#Cargar los datos
df_train = pd.read_csv('entrenamiento.csv', index_col='ID')
df_test = pd.read_csv('prueba.csv', index_col='ID')

atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']

# Pasamos los datos a formato Numpy (matrices) para poder hacer matemáticas
X_train = df_train[atributos].values
X_test = df_test[atributos].values

# Para calcular el coeficiente independiente (b0),
# necesitamos agregar una columna llena de unos (1) al principio de nuestros datos.
X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]


# La fórmula Sigmoide que convierte números Z en probabilidades de 0 a 1
def sigmoide(z):
    z = np.clip(z, -250, 250)  # Clip evita que la computadora colapse con números gigantes
    return 1 / (1 + np.exp(-z))


# El reemplazo manual de modelo.fit()
def entrenar_manualmente(X, y, tasa_aprendizaje=0.005, iteraciones=10000):
    pesos = np.zeros(X.shape[1])  # b0, b1, b2... empiezan en cero
    total_datos = len(y)

    for i in range(iteraciones):
        # 1. Multiplicar atributos por pesos (Z) y aplicar sigmoide
        z = np.dot(X, pesos)
        predicciones = sigmoide(z)

        # 2. Calcular qué tan equivocados estamos (Error)
        error = predicciones - y

        # 3. Ajustar los pesos usando el Gradiente
        gradiente = np.dot(X.T, error) / total_datos
        pesos = pesos - (tasa_aprendizaje * gradiente)

    return pesos  # Devuelve los coeficientes b0 a b5 ya entrenados


#Aplicamos el motor a las clases

clases_rpg = ['Guerrero', 'Pícaro', 'Mago', 'Guardián']
pesos_guardados = {}  # Aquí guardaremos los coeficientes (b) de cada clase

print("Entrenamiento manual \n")
for clase in clases_rpg:
    # 1 si es la clase, 0 si no
    y_train = (df_train['Clase'] == clase).astype(int).values
    y_test = (df_test['Clase'] == clase).astype(int).values

    # Entrenar
    pesos = entrenar_manualmente(X_train_b, y_train)
    pesos_guardados[clase] = pesos  # Guardar b0..b5 para usarlos luego

    # Evaluar precisión
    probabilidades_test = sigmoide(np.dot(X_test_b, pesos))
    predicciones_test = (probabilidades_test >= 0.5).astype(int)

    # Comparamos nuestras predicciones con la realidad
    precision = np.mean(predicciones_test == y_test)
    print(f"Modelo Manual: ¿Es {clase}? -> Precisión: {precision * 100:.1f}%")


# Personaje: Fuerza 89, Destreza 13, Inteligencia 26, Constitución 8, Suerte 78
nuevo_personaje = np.array([89, 13, 26, 8, 78])
# Le agregamos el '1' al principio para que multiplique al b0
nuevo_personaje_b = np.insert(nuevo_personaje, 0, 1)

print("\n")
for clase in clases_rpg:
    # Recuperamos los coeficientes que la computadora aprendió para esta clase
    pesos_clase = pesos_guardados[clase]

    # Calculamos Z multiplicando los atributos del personaje por los pesos
    z = np.dot(nuevo_personaje_b, pesos_clase)

    # Pasamos Z por la función sigmoide para obtener el porcentaje
    prob_si_es_clase = sigmoide(z)

    print(f"Probabilidad de ser {clase}: {prob_si_es_clase * 100:.2f}%")