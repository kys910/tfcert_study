import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

#Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_valid = x_valid / 255.0

tf.keras.backend.set_floatx('float64')

x = Flatten(input_shape=(28, 28))
print(x(x_train).shape)

model = Sequential([
    # Flatten으로 shape 펼치기
    Flatten(input_shape=(28, 28)),
    # Dense Layer
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    # Classification을 위한 Softmax
    Dense(10, activation='softmax'),
])

model.summary()

print(y_train[0])

print(tf.one_hot(y_train[0], 10))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])