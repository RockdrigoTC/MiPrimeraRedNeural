import tensorflow as tf
import pandas as pd

# Cargar los datos de prueba desde un archivo CSV
x_test = pd.read_csv('x_test.csv')

# Crear la red neuronal secuencial
model = tf.keras.Sequential()

# Agregar capas a la red neuronal
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Cargar los pesos del modelo previamente entrenado
model.load_weights('model_weights.h5')

# Predecir las salidas para los datos de prueba
y_pred = model.predict(x_test)


print("Resultados de la predicción:")
print(y_pred)
print("Resultados de la predicción redondeados:")
print(y_pred.round().astype(int))
