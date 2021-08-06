# 2. 모델 불러오기
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

model = load_model('TF2-mnist.h5')

# 3. 모델 사용하기
print(model.evaluate(x_valid, y_valid))

