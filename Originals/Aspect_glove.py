
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

#get TF
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys


# Importing texts

import xml.etree.ElementTree as ET
print('Processing text dataset')

tree = ET.parse("../Restaurants_Train.xml")
corpus = tree.getroot()
sentences = [] # List of list of sentences.
sent = corpus.findall('.//sentence')
for s in sent:
    sentences.append(s.find('text').text)
#sentences has all sentence test
print ('Generated list of sentences..')


MAX_SEQ_LENGTH = 69
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300


# Indexing Word Vectors
print('Indexing word vectors.')

embeddings_index = {}
f = open('glove.6B\glove.6B.300d.txt',encoding='UTF8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# vectorize the text samples into a 2D integer tensor and padding the sentences


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, lower=False)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print ("Let's have a quick look at the word_index data..")
print (list(word_index.items())[:10])

data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
print('Shape of data tensor:', data.shape)


# defining output data
import nltk
from keras.preprocessing.text import text_to_word_sequence
raw_output = corpus.findall('.//sentence')
train_out= np.zeros(shape=(3044,69))
i=0
for output in raw_output:
    s = text_to_word_sequence(output.find('text').text, lower=False)
    indices = np.zeros(MAX_SEQ_LENGTH)
    aspectTerms = output.find('aspectTerms')
    if (aspectTerms):
        aspectTerm = aspectTerms.findall('aspectTerm')
        if (aspectTerm):
            for aspect_term in aspectTerm:
                try:
                    indices[s.index(aspect_term.attrib['term'])] = 1
                except:
                    continue
    train_out[i] = indices
    i=i+1

print (train_out.shape)

# #### Preparing Embedding Layer
print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words + 1, 300))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False)


# Defining and Training Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing import sequence
from keras.optimizers import *
from keras.regularizers import l2

print('Training model.')

model = Sequential()
model.add(embedding_layer)
model.add(Convolution1D(100, 5, border_mode="same", input_shape=(65, 300)))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_length=5))
model.add(Convolution1D(50, 3, border_mode="same"))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("tanh"))
# softmax classifier
model.add(Dense(69, W_regularizer=l2(0.01)))
model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


print (model.summary())

model.fit(data, train_out,
          validation_split=0.1,
          batch_size=10,
          nb_epoch=50
         )

model.save_weights('aspect_model.h5')

test_data = data[:10]
test_output = train_out[:10]

output = model.predict(test_data,verbose=1)

print (output[1])

