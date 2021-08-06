#필요 모듈 import
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

#Dataset 로드
#가위바위보에 대한 손의 사진을 가지고 가위인지, 바위인지, 보자기인지 분류하는 classification 문제입니다.
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

#STEP 2. Define Folder
#데이터셋의 경로를 지정해 주세요 (root 폴더의 경로를 지정하여야 합니다.)
#training dir
TRAINING_DIR = 'tmp/rps/'

#STEP 3. ImageDataGenerator
#rescale: 이미지의 픽셀 값을 조정
#rotation_range: 이미지 회전
#width_shift_range: 가로 방향으로 이동
#height_shift_range: 세로 방향으로 이동
#shear_range: 이미지 굴절
#zoom_range: 이미지 확대
#horizontal_flip: 횡 방향으로 이미지 반전
#fill_mode: 이미지를 이동이나 굴절시켰을 때 빈 픽셀 값에 대하여 값을 채우는 방식
#validation_split: train set / validation set 분할 비율
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

#STEP 4. Make Generator
#ImageDataGenerator를 잘 만들어 주었다면, flow_from_directory로 이미지를 어떻게 공급해 줄 것인가를 지정해 주어야합니다.

#train / validation set 전용 generator를 별도로 정의합니다.
#batch_size를 정의합니다 (128)
#target_size를 정의합니다. (150 x 150). 이미지를 알아서 타겟 사이즈 만큼 잘라내어 공급합니다.
#class_mode는 3개 이상의 클래스인 경우 'categorical' 이진 분류의 경우 binary를 지정합니다.
#subset을 지정합니다. (training / validation)

#training_generator에 대한 from_from_directory를 정의합니다.
training_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                          batch_size=32,
                                                          target_size=(150,150),
                                                          class_mode='categorical',
                                                          subset='training',
                                                          )
#validation_generator에 대한 from_from_directory를 정의합니다.
validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                            batch_size=32,
                                                            target_size=(150, 150),
                                                            class_mode='categorical',
                                                            subset='validation',
                                                            )

#시각화 해보기 - 시험에 안나오는 코드임

class_map = {
    0: 'Paper',
    1: 'Rock',
    2: 'Scissors'
}

print('오리지널 사진 파일')

original_datagen = ImageDataGenerator(rescale=1./255)
original_generator = original_datagen.flow_from_directory(TRAINING_DIR,
                                                          batch_size=32,
                                                          target_size=(150, 150),
                                                          class_mode='categorical'
                                                         )

for x, y in original_generator:
    print(x.shape, y.shape)
    print(y[0])

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 6)
    for i in range(10):
        axes[i // 5, i % 5].imshow(x[i])
        axes[i // 5, i % 5].set_title(class_map[y[i].argmax()], fontsize=15)
        axes[i // 5, i % 5].axis('off')
    plt.show()
    break

print('Augmentation 적용한 사진 파일')

for x, y in training_generator:
    print(x.shape, y.shape)
    print(y[0])

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 6)
    for i in range(10):
        axes[i // 5, i % 5].imshow(x[i])
        axes[i // 5, i % 5].set_title(class_map[y[i].argmax()], fontsize=15)
        axes[i // 5, i % 5].axis('off')

    plt.show()
    break

#Convolution Neural Network (CNN)
#CNN - activation - Pooling 과정을 통해 이미지 부분 부분의 주요한 Feature 들을 추출해 냅니다.
#CNN을 통해 우리는 다양한 1개의 이미지를 filter를 거친 다수의 이미지로 출력합니다.
#filter의 사이즈는 3 X 3 필터를 자주 사용합니다
#또한, 3 X 3 필터를 거친 이미지의 사이즈는 2px 만큼 사이즈가 줄어듭니다.
#Image('https://devblogs.nvidia.com/wp-content/uploads/2015/11/fig1.png', width=800)

#이미지 특성 추출:Conv2D
for x, y in original_generator:
    pic = x[:5]
    break

plt.imshow(pic[0])

conv2d = Conv2D(64, (3, 3), input_shape=(150, 150, 3))
conv2d_activation = Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3))

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(16, 16)
for i in range(64):
    axes[i//8, i % 8].imshow(conv2d(pic)[0,:,:,i], cmap='gray')
    axes[i//8, i % 8].axis('off')

#이미지 특성 추출: MaxPooling2D
fig, axes = plt.subplots(8, 8)
fig.set_size_inches(16, 16)
for i in range(64):
    axes[i // 8, i % 8].imshow(MaxPooling2D(2, 2)(conv2d(pic))[0,:,:,i], cmap='gray')
    axes[i // 8, i % 8].axis('off')

#단계별 특성 추출 과정
conv1 = Conv2D(64, (3, 3), input_shape=(150, 150, 3))(pic)
max1 = MaxPooling2D(2, 2)(conv1)
conv2 = Conv2D(64, (3, 3))(max1)
max2 = MaxPooling2D(2, 2)(conv2)
conv3 = Conv2D(64, (3, 3))(max2)
max3 = MaxPooling2D(2,2)(conv3)

fig, axes = plt.subplots(4, 1)
fig.set_size_inches(6, 12)
axes[0].set_title('Original', fontsize=20)
axes[0].imshow(pic[0])
axes[0].axis('off')
axes[1].set_title('Round 1', fontsize=20)
axes[1].imshow(conv1[0,:,:,1], cmap='gray')
axes[1].axis('off')
axes[2].set_title('Round 2', fontsize=20)
axes[2].imshow(conv2[0,:,:,1], cmap='gray')
axes[2].axis('off')
axes[3].set_title('Round 3', fontsize=20)
axes[3].imshow(conv3[0,:,:,1], cmap='gray')
axes[3].axis('off')
plt.tight_layout()
plt.show()

#모델 정의 (Sequential)
model = Sequential([
    #Conv3D, MaxPooling2D 조합으로 층을 쌓습니다. 첫번째 입력층의 input_shape은(150, 150, 3)으로 지정합니다.
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    #2D -> 1D로 변환을 위하여 Flatten 합니다.
    Flatten(),
    #과적합 방지를 위하여 Dropout을 적용합니다.
    Dropout(0.5),
    Dense(512, activation='relu'),
    #Classification을 위한 softmax
    #출력층의 갯수는 클래스의 갯수와 동일하게 맞춰줍니다. (3개) activation도 잊지마세요!
    Dense(3, activation='softmax')
])

model.summary()

#컴파일 (compile)
#optimizer는 가장 최적화가 잘되는 알고리즘인 'adam'을 사용합니다.
#loss설정
#출력층 activation이 sigmoid 인 경우: binary_crossentropy
#출력층 activation이 softmax 인 경우:
#원핫인코딩(O): categorical_crossentropy
#원핫인코딩(X): sparse_categorical_crossentropy)
#참고: ImageDataGenerator는 자동으로 Label을 원핫인코딩(one-hot encoding) 해줍니다.
#metrics를 'acc' 혹은 'accuracy'로 지정하면, 학습시 정확도를 모니터링 할 수 있습니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#ModelCheckpoint: 체크포인트 생성
#val_loss 기준으로 epoch 마다 최적의 모델을 저장하기 위하여, ModelCheckpoint를 만듭니다.
#checkpoint_path는 모델이 저장될 파일 명을 설정합니다.
#ModelCheckpoint을 선언하고, 적절한 옵션 값을 지정합니다.
checkpoint_path = 'tmp_checkpoint.ckpt'
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

#학습 fit
epochs = 25
history = model.fit(training_generator,
                    validation_data=(validation_generator),
                    epochs=epochs,
                    callbacks=[checkpoint],
                    )

#습 완료 후 Load Weights (ModelCheckpoint)
#학습이 완료된 후에는 반드시 load_weights를 해주어야 합니다.
#그렇지 않으면, 열심히 ModelCheckpoint를 만든 의미가 없습니다.
model.load_weights(checkpoint_path)

#학습 오차에 대한 시각화
plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, epochs+1), history.history['acc'])
plt.plot(np.arange(1, epochs+1), history.history['loss'])
plt.title('Acc / Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Acc / Loss')
plt.legend(['acc', 'loss'], fontsize=15)
plt.show()