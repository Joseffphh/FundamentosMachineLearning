import pandas as pd
import numpy as np

# 1. Leer el archivo de entrenamiento
df_train = pd.read_csv('entrenamiento.csv', index_col='ID')

# 2. Crear una nueva columna binaria sin borrar la original
df_train['Es_Guerrero'] = np.where(df_train['Clase'] == 'Guerrero', 'Sí', 'No')

# 3. Filtrar
guerreros = df_train.query("Es_Guerrero == 'Sí'")
otras_clases = df_train.query("Es_Guerrero == 'No'")

# Mostrar resultados
print("--- Todos los personajes (con nueva columna) ---")
print(df_train.head())

print("\n--- Solo Guerreros ---")
print(guerreros)