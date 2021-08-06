'''
Create a classifier for the Fashion MNIST dataset

Note that the test will expect it to classify 10 classes and that

the input shape should be the native size of the Fashion MNIST dataset which is 28x28 monochrome.

Do not resize the data. Your input layer should accept

(28,28) as the input shape only.

If you amend this, the tests will fail.

Fashion MNIST 데이터 셋에 대한 분류기 생성 테스트는 10 개의 클래스를 분류 할 것으로 예상하고

입력 모양은 Fashion MNIST 데이터 세트의 기본 크기 여야합니다.28x28 단색.

데이터 크기를 조정하지 마십시오. input_shape는 (28,28)을 입력 모양으로 만 사용합니다.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

#Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

print(x_train.shape)
print(x_valid.shape)

print(y_train.shape)
print(y_valid.shape)

print(x_train.min())
print(x_valid.max())

#정규화 실행
x_train = x_train / 255.0
x_valid = x_valid / 255.0

#정규화 실행 후 확인
print(x_train.min())
print(x_valid.max())


# 시각화
fig, axes = plt.subplots(2, 5)
fig.set_size_inches(10, 5)

'''
사진 예시 확인
for i in range(10):
    axes[i//5, i%5].imshow(x_train[i], cmap='gray')
    axes[i//5, i%5].set_title(str(y_train[i]), fontsize=15)
    plt.setp( axes[i//5, i%5].get_xticklabels(), visible=False)
    plt.setp( axes[i//5, i%5].get_yticklabels(), visible=False)
    axes[i//5, i%5].axis('off')

plt.tight_layout()
plt.show()
'''
'''
Flatten이란?

고차원을 1D로 변환하여 Dense Layer에 전달해 주기 위하여 사용합니다.
28 X 28 의 2D로 되어 있는 이미지를 784로 1D로 펼쳐 주는 작업입니다.
'''

tf.keras.backend.set_floatx('float64')

print(x_train.shape)

x = Flatten(input_shape=(28, 28))
print(x(x_train).shape)

from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

'''
def relu(x):
    return np.maximum(x, 0)

x = np.linspace(-10, 10)
y = relu(x)

plt.figure(figsize=(10, 7))
plt.plot(x, y)
plt.title('ReLU activation function')
plt.show()
'''

'''
이제 Modeling을 할 차례입니다.

Sequential 모델 안에서 층을 깊게 쌓아 올려 주면 됩니다.

Dense 레이어는 2D 형태의 데이터를 받아들이지 못합니다. Flatten 레이어로 2D -> 1D로 변환해주세요
깊은 출력층과 더 많은 Layer를 쌓습니다.
Dense Layer에 activation='relu'를 적용합니다.
분류(Classification)의 마지막 층의 출력 숫자는 분류하고자 하는 클래스 갯수와 같아야 합니다.
'''

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

checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=20,
                    callbacks=[checkpoint],
                   )

# checkpoint 를 저장한 파일명을 입력합니다.
model.load_weights(checkpoint_path)

model.evaluate(x_valid, y_valid)

print(model.predict(x_valid[:1]))


'''
plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['loss'])
plt.plot(np.arange(1, 21), history.history['val_loss'])
plt.title('Loss / Val Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'], fontsize=15)
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['acc'])
plt.plot(np.arange(1, 21), history.history['val_acc'])
plt.title('Acc / Val Acc', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'], fontsize=15)
plt.show()
'''