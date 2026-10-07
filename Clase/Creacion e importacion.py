import pandas as pd
import numpy as np

# 1. Crear los datos base
np.random.seed(42)
atributos = ['Fuerza', 'Destreza', 'Inteligencia', 'Constitución', 'Suerte']
df = pd.DataFrame(np.random.randint(0, 101, size=(50, 5)), columns=atributos)
df.index += 1
df.index.name = 'ID'

# 2. Asignar clase usando idxmax (Encuentra la columna con el valor máximo)
mapa_clases = {
    'Fuerza': 'Guerrero',
    'Destreza': 'Pícaro',
    'Inteligencia': 'Mago',
    'Constitución': 'Guardián'
}
# Solo evaluamos las 4 estadísticas principales (ignoramos Suerte)
df['Clase'] = df[['Fuerza', 'Destreza', 'Inteligencia', 'Constitución']].idxmax(axis=1).map(mapa_clases)

# 3. Dividir y exportar en una sola línea por archivo
df.iloc[:30].to_csv('entrenamiento.csv')
df.iloc[30:].to_csv('prueba.csv')

print("Archivos exportados con éxito.")