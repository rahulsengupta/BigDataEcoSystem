from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN


# set parameters:
batch_size = 16
input_dim = 1
nb_epoch = 50

print('Loading data...')
file1 = "training.txt"
file2 = "testing.txt"

# data loading
data1 = np.loadtxt(file1, delimiter=',')
X_train = data1[:50000, :500, newaxis]
print(X_train.shape)

y_train = data1[:50000, -1]
print(y_train.shape)

data2 = np.loadtxt(file2, delimiter=',')
X_test = data2[:, :500, newaxis]
print(X_test.shape)


y_test = data2[:, -1]
print(y_test.shape)

X_train = X_train[0:20000]
y_train = y_train[0:20000]
X_test = X_test[0:5000]
y_test = y_test[0:5000]

print('Build model...')
model = Sequential()

model.add(LSTM(output_dim=64, input_dim=input_dim))
model.add(Dropout(0.25))
model.add(Activation('relu'))
#two FC layers
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

print('Train...')
modelprints = model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

print('Build model2...')
model2 = Sequential()

model2.add(GRU(output_dim=64, input_dim=input_dim))
model2.add(Dropout(0.25))
model2.add(Activation('relu'))
#two FC layers
model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(Dropout(0.25))
model2.add(Dense(128))
model2.add(Dropout(0.25))
model2.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model2.add(Dense(1)) 
model2.add(Activation('sigmoid'))

model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

print('Train...')
model2prints = model2.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
score, acc = model2.evaluate(X_test, y_test, batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

print(modelprints)
print(model2prints)
