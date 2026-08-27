import pandas as pd

#Lectura del archivo
df = pd.read_csv('dataset_musica_generado.csv')

#Conversion a segundos
def tiempoSegundos(tiempo_str):
    #Funcion .split toma una cadena de texto y la corta en pedazos usando un caracter, este caso es :
    #Funcion map convierte a un numero entero, en este caso
    #minutos y segundos son variables para los numeros
    minutos, segundos = map(int, tiempo_str.split(':'))
    #Se hace la conversion matematica, multiplicando los minutos por 60, y sumando los segundos
    return minutos * 60 + segundos

df['Duracion_segundos'] = df['Duración'].apply(tiempoSegundos)

#Analisis descriptivo
#La funcion .describe() agrupa todas las metricas para columnas numericas.
estadisticas = df[['Reproducciones', 'Duracion_segundos']].describe().round(2)

print('estadisticas')
print(estadisticas)

#Calculo de la moda
moda_genero = df['Género'].mode()[0]
moda_artista = df['Artista'].mode()[0]

print("Modas")
print(f"Genero mas común: {moda_genero}")
print(f"Artista mas común: {moda_artista}")
