# import tensorflow as tf
# import numpy as np
# import pandas as pd

# # Cargar los datos de entrada y salida desde los archivos CSV
# x_in = pd.read_csv('x_in.csv')
# y_out = pd.read_csv('y_out.csv')

# # Crear la red neuronal secuencial
# model = tf.keras.Sequential()

# # Agregar capas a la red neuronal
# model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# # Compilar la red neuronal
# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# # Entrenar la red neuronal solo si no se ha entrenado antes
# try:
#     model.load_weights('model_weights.h5')
#     print('Modelo cargado desde el archivo')
# except:
#     print('Creando modelo...')

# print('Entrenando modelo...')
# model.fit(x_in, y_out, epochs=3000)
# model.save_weights('model_weights.h5')
# print('Modelo guardado en el archivo')

# # Evaluar la red neuronal
# x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_test = np.array([0, 1, 1, 0])
# loss, accuracy = model.evaluate(x_test, y_test)

# print(f'Pérdida: {loss}, Precisión: {accuracy}')

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Cargar los datos de entrada y salida desde los archivos CSV
x_in = pd.read_csv('x_in.csv')
y_out = pd.read_csv('y_out.csv')

# Crear la red neuronal secuencial
model = tf.keras.Sequential()

# Agregar capas a la red neuronal
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compilar la red neuronal
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Entrenar la red neuronal solo si no se ha entrenado antes
try:
    model.load_weights('model_weights.h5')
    print('Modelo cargado desde el archivo')
except:
    print('Creando modelo...')

print('Entrenando modelo...')
for i in tqdm(range(3000)):
    model.train_on_batch(x_in, y_out)
model.save_weights('model_weights.h5')
print('Modelo guardado en el archivo')

# Evaluar la red neuronal
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([0, 1, 1, 0])
loss, accuracy = model.evaluate(x_test, y_test)

print(f'Pérdida: {loss}, Precisión: {accuracy}')
