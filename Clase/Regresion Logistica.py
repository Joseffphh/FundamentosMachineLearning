import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Cargar los datos
df_train = pd.read_csv('entrenamiento.csv', index_col='ID')
df_test = pd.read_csv('prueba.csv', index_col='ID')

# 2. Separar los atributos (X) multivariados
atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']
X_train = df_train[atributos]
X_test = df_test[atributos]

# Las clases que queremos evaluar
clases_rpg = ['Guerrero', 'Pícaro', 'Mago', 'Guardián']

# 3. Entrenar y evaluar un modelo para cada clase
for clase in clases_rpg:
    # Crear variable objetivo (Y) binaria: 1 si es la clase, 0 si no lo es
    y_train = (df_train['Clase'] == clase).astype(int)
    y_test = (df_test['Clase'] == clase).astype(int)

    # Iniciar y entrenar el modelo de Regresión Logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Calcular la precisión usando los 20 registros de prueba
    precision = modelo.score(X_test, y_test)

    # Mostrar resultados
    print(f"Modelo: ¿Es {clase}?")
    print(f"Precisión: {precision * 100:.1f}%\n")

    # (Asumiendo que ya tienes cargados df_train y X_train como en el paso anterior)
    atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']
    clases_rpg = ['Guerrero', 'Pícaro', 'Mago', 'Guardián']

    # 1. Crear un "nuevo personaje" para evaluar (o tomar uno de tus datos de prueba)
    # Este personaje tiene: Fuerza alta (89), y
    nuevo_personaje = pd.DataFrame(
        [[89, 13, 26, 8, 78]],
        columns=atributos
    )

    print("Evaluando al personaje:")
    print(nuevo_personaje.to_markdown(), "\n")

    # 2. Iterar por cada clase para ver qué opina cada modelo
    for clase in clases_rpg:
        # Entrenar el modelo (como lo hicimos antes)
        y_train = (df_train['Clase'] == clase).astype(int)
        modelo = LogisticRegression()
        modelo.fit(X_train, y_train)

        # 3. Calcular el valor proyectado (Probabilidad)
        # predict_proba devuelve una matriz con 2 valores: [Prob de NO ser, Prob de SI ser]
        probabilidad_completa = modelo.predict_proba(nuevo_personaje)

        # Extraemos solo el segundo valor (índice 1), que es la probabilidad de que SÍ sea de la clase
        prob_si_es_clase = probabilidad_completa[0][1]

        # Mostrar el porcentaje
        print(f"Probabilidad de ser {clase}: {prob_si_es_clase * 100:.2f}%")