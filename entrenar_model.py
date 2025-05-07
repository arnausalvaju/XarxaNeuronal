from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configuració de logger per evitar missatges d’error extensos
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Carregar el conjunt de dades MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
class_names = ['Zero', 'U', 'Dos', 'Tres', 'Quatre', 'Cinc', 'Sis', 'Set', 'Vuit', 'Nou']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print(f"Exemples d'entrenament: {num_train_examples}")
print(f"Exemples de prova: {num_test_examples}")

# Normalització: valors de píxel de 0-255 a 0-1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Estructura de la xarxa neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")  # Sortida per classificació de dígits
])

# Compilar el model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenament per lots
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

# Entrenar el model
print("Iniciant entrenament...")
model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE)
)

# Avaluar el model entrenat amb el conjunt de prova
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32)
)
print("Precisió sobre les dades de prova:", test_accuracy)

# Desar el model entrenat
model.save("modelo_mnist.keras")
print("Model desat com a 'modelo_mnist.keras'")

print("Entrenament completat! Ara pots executar el servidor.")