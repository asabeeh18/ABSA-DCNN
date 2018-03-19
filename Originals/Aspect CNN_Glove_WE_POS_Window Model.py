
# coding: utf-8

# ### Importing Dependencies

# In[1]:


from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys


# ### Importing Text Data

# In[2]:


import xml.etree.ElementTree as ET
print('Processing text dataset')

tree = ET.parse("..\Restaurants_Train.xml")
corpus = tree.getroot()
sentences = [] # List of list of sentences.
sent = corpus.findall('.//sentence')
for s in sent:
    sentences.append(s.find('text').text)

print ('Generated list of sentences..')

MAX_SEQ_LENGTH = 69
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300


# ### Indexing Word Vectors

# In[3]:


print('Indexing word vectors.')

embeddings_index = {}
f = open('glove.6B/glove.6B.300d.txt',encoding='UTF8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# ### Vectorize the sentences into 2D Tensors and Padding

# In[4]:


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, lower=False)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Here padding has been done at both the ends since we will need to take the context window size of 4 units.
data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH+2, padding='post')
data = pad_sequences(data, maxlen=MAX_SEQ_LENGTH+4, padding='pre')
print('Shape of data tensor:', data.shape)


# ### Defining Output Data

# In[5]:


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
#                     print (indices)
                except:
                    continue
    train_out[i] = indices
    i=i+1

print ("Shape of output tensor:", train_out.shape)


# ### Preparing Embedding Layer

# In[6]:


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
# Here, we have set trainable = False so as to keep the embeddings fixed.
embedding_layer = Embedding(nb_words + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH+4,
                            trainable=False)
print('Embedding Layer set..')


# ### Set Word Embeddings

# In[7]:


from keras.models import Sequential
embedding_model = Sequential()
embedding_model.add(embedding_layer)

embedding_model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['acc']
                       )
embedding_output = embedding_model.predict(data)
print('Generated word Embeddings..')
print('Shape of Embedding_output', embedding_output.shape)


# ### Adding POS Tag Features

# In[11]:


from keras.preprocessing.text import text_to_word_sequence
from nltk.tag.stanford import StanfordPOSTagger
from sklearn import preprocessing
from tqdm import tqdm

train_input = np.zeros(shape=(3044,69,306))
le = preprocessing.LabelEncoder()
tags = ["CC","NN","JJ","VB","RB","IN"]
le.fit(tags)
i=0
sentences = corpus.findall('.//sentence')
for sent in sentences:
    s = text_to_word_sequence(sent.find('text').text)
    tags_for_sent = nltk.pos_tag(s)
    sent_len = len(tags_for_sent)
    ohe = [0]*6

    for j in range(69):
        if j< len(tags_for_sent) and tags_for_sent[j][1][:2] in tags:
            ohe[le.transform(tags_for_sent[j][1][:2])] = 1
        train_input[i][j] = np.concatenate([embedding_output[i][j+2],ohe])
    i=i+1
    
print('Concatenated Word-Embeddings and POS Tag Features..')


# ### WE + POS + Window Feature Model

# In[10]:


from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing import sequence
from keras.optimizers import *
from keras.regularizers import l2


# In[29]:


model = Sequential()
model.add(Convolution1D(100, 5, border_mode="same", input_shape=(69, 306), input_length=5))
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

# model.load_weights('aspect_model_wepos.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print('Model Trained..')


# In[30]:


model.fit(train_input, train_out,
          validation_split=0.1,
          batch_size=10,
          nb_epoch=50
         )


# In[31]:


model.save_weights('aspect_wepos_window.h5')
y_pred = model.predict(train_input[2739:])


# ### Evaluating Window Model

# In[48]:


processed_output = []
for i in xrange(y_pred.shape[0]):
    processed_label =[]
    for j in xrange(y_pred.shape[1]):
        if y_pred[i][j] > 0.42:
            processed_label.append(1)
        else:
            processed_label.append(0)
    processed_output.append(processed_label)


# In[49]:


test_data = train_out[2739:]
total_pos = 0.0
true_pos = 0.0
total_neg = 0.0
true_neg = 0.0
for i in xrange(test_data.shape[0]):
    for j in xrange(test_data.shape[1]):
        if test_data[i][j] == 1:
            total_pos += 1
            if processed_output[i][j] ==1:
                true_pos +=1
        if test_data[i][j] == 0:
            total_neg += 1
            if processed_output[i][j] ==0:
                true_neg += 1

false_pos = total_neg-true_neg
false_neg = total_pos-true_pos
precision = true_pos/(true_pos+false_pos)
recall = true_pos/total_pos


# In[50]:


f1_score = 2*precision*recall/(precision+recall)
print ("precision- " +str(precision) + ", recall- " +str(recall)+ ",f1_score- " +str(f1_score))

