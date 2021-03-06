from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

from random import shuffle
import numpy as np
import sys
import cPickle

path = sys.argv[1]
trainX = []
trainY = []
print ("[*] Preprocessing data")
with open(path, 'r') as f:
    counter = 0
    lines = f.readlines()
    for line in lines:
        if counter % 10000 == 0:
            print("[*] %s" % counter)
        trainY.append(int(line[0]))
        sp = [int(tok) for tok in line[2:].split()]
        trainX.append(sp)
        counter += 1
print ("[*] Data util")
if not len(trainX) == len(trainY):
    print ("[*] LenError: len(trainX) should equal to len(trainY)")
    sys.exit(-1)
indexShuffle = range(len(trainX))
shuffle(indexShuffle)
trainX = [trainX[i] for i in indexShuffle]
trainY = [trainY[i] for i in indexShuffle]
# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=200, value=0.)
#testX = pad_sequences(testX, maxlen=200, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
#valY = to_categorical(valY, nb_classes=2)
#testY = to_categorical(testY, nb_classes=2)

print ("[*] X = %s" % trainX)
print ("[*] Building net")
# Network building
net = tflearn.input_data([None, 200])
net = tflearn.embedding(net, input_dim=500000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                                 loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=0.3, show_metric=True,
                  batch_size=128)
'''
# Network building
net = input_data(shape=[None, 200])
net = embedding(net, input_dim=500000, output_dim=128)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = dropout(net, 0.5)
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

print ("[*] Training model")

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.3, show_metric=True, batch_size=64)
'''
print ("[*] Saving model")

with open('model_bilstm.pkl', 'wb') as f:
    cPickle.dump(model, f)
