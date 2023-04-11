import tensorflow as tf
import numpy as np
import pandas as pd

# Cargar los datos de prueba desde un archivo CSV
# x_test = pd.read_csv('x_test.csv')
x_test = pd.read_csv('x_test.csv', header=0, delimiter=',')

# Crear la red neuronal secuencial
model = tf.keras.Sequential()

# Agregar capas a la red neuronal
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Cargar los pesos del modelo previamente entrenado
model.load_weights('model_weights.h5')

# Predecir las salidas para los datos de prueba
y_pred = model.predict(x_test)

print(y_pred)
