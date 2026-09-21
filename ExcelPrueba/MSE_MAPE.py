#datos finales en array.
y_real = [52.0, 56.0, 59.0, 64.0, 67.0, 72.0]
y_pred = [51.8095, 55.7523, 59.6952, 63.6380, 67.5809, 71.5238]

#variables para ir sumando los errores
suma_error_absoluto = 0
suma_error_porcentual = 0
#len retorna el numero de items(los valores).
n = len(y_real)

#repasamos dato por dato usando un ciclo
for i in range(n):
    real = y_real[i]
    pred = y_pred[i]
#calculamos la diferencia y la volvemos positiva con abs(); abs retorna el valor absoluto.
    error_absoluto = abs(real - pred)
    suma_error_absoluto += error_absoluto
#calculamos la diferencia porcentual
    error_porcentual = abs((real - pred) / real)
    suma_error_porcentual += error_porcentual

#promediamos y convertimos el MAPE a porcentaje final
    mae = suma_error_absoluto / n
    mape = (suma_error_porcentual / n) * 100

print(f"MAE calculado: {mae}")
print(f"MAPE calculado: {mape}%")