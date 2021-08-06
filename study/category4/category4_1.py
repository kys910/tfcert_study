import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

with open('sarcasm.json') as f:
    datas = json.load(f)

#print(datas[:5])

sentences = []
labels = []

for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])

print(sentences[:5])

print(labels[:5])

training_size = 20000

train_sentences = sentences[:training_size]
train_labels = labels[:training_size]

validation_sentences = sentences[training_size:]
validation_labels = labels[training_size:]

vocab_size = 1000
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

tokenizer.fit_on_texts(train_sentences)

for key, value in tokenizer.word_index.items():
    print('{}  \t======>\t {}'.format(key, value))
    if value == 25:
        break

print(len(tokenizer.word_index))
word_index = tokenizer.word_index
word_index['trump']
word_index['hello']
word_index['<OOV>']


train_sequences = tokenizer.texts_to_sequences(train_sentences)
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

train_sequences[:5]
train_sentences[4]
word_index['j'], word_index['k'], word_index['rowling'], word_index['wishes'], word_index['snape'], word_index['happy']
train_sequences[4]

# 한 문장의 최대 단어 숫자
max_length = 120

# 잘라낼 문장의 위치
trunc_type='post'

# 채워줄 문장의 위치
padding_type='post'

train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_padded.shape

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

embedding_dim = 16

sample = np.array(train_padded[0])
sample

x = Embedding(vocab_size, embedding_dim, input_length=max_length)
x(sample)[0]

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

epochs=10

history = model.fit(train_padded, train_labels,
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)


model.load_weights(checkpoint_path)