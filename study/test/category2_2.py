import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

for data in train_dataset.take(1):
    print(data)


    def preprocess(data):
        # 코드를 입력하세요
        x = data['features']
        y = data['label']
        y = tf.one_hot(y, 3)
        return x, y