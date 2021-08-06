# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# This task requires you to create a classifier for horses or humans using
# the provided dataset.
#
# Please make sure your final layer has 2 neurons, activated by softmax
# as shown. Do not change the provided output layer, or tests may fail.
#
# IMPORTANT: Please note that the test uses images that are 300x300 with
# 3 bytes color depth so be sure to design your input layer to accept
# these, or the tests will fail.
#

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - Horses Or Humans type A
# val_loss: 0.028
# val_acc: 0.98
# =================================================== #
# =================================================== #


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import urllib.request
import zipfile
import numpy as np
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'horses_or_humans'
#dataset, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)
train_ds, test_ds = tfds.load(name=dataset_name, split=['train', 'test'])

'''
for data in train_ds.take(1):
    image, label = data['image'], data['label']
    print(image)
    print(label)
'''

def preprocess(data):
    x = data['image']
    y = data['label']
    # image 정규화(Normalization)
    x = x / 255
    # 사이즈를 (224, 224)로 변환합니다.
    x = tf.image.resize(x, size=(224, 224))
    return x, y


def solution_model():
    #train_dataset = dataset.map(preprocess).batch(32)

    train_data = train_ds.map(preprocess).batch(32)
    valid_data = test_ds.map(preprocess).batch(32)

    training_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
    )

    validation_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
    )

    '''
        train_generator = training_datagen.flow_from_directory(
            TRAINING_DIR,
            target_size=(300, 300),
            batch_size=32,
            class_mode='categorical',
        )
    
        validation_generator = validation_datagen.flow_from_directory(
            VALIDATION_DIR,
            target_size=(300, 300),
            batch_size=32,
            class_mode='categorical',
        )
    '''

    model = Sequential([
        # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
        Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(16, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        # Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    print(model.summary())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "tmp_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_data,
              validation_data=(valid_data),
              epochs=100,
              callbacks=[checkpoint],
              )

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    #model.save("TF3-horses-or-humans-type-A.h5")
