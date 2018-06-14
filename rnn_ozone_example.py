import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
import requests

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.utils import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

plt.style.use('bmh')

names = ['feat_{}'.format(i) for i in range(73)]
df = pd.read_csv('./dataset/eighthr.data', names=names)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

Y = df['feat_72']
Y = to_categorical(Y)
df.drop(['feat_72'], axis=1, inplace=True)
Y_train = Y[:-100]
Y_test = Y[-100:]
# print('Y_train.shape:', Y_train.shape, ', Y_test.shape:', Y_test.shape)

X_train = np.array(df[:-100].values.tolist(), dtype=np.float64)
X_test = np.array(df[-100:].values.tolist(), dtype=np.float64)
# print('X_train.shape:', X_train.shape, ', X_test.shape:', X_test.shape)

X_train = X_train[:1700]
Y_train = Y_train[:1700]
# print('X_train.shape:', X_train.shape, ', Y_train.shape:', Y_train.shape)

X_train = X_train.reshape(-1, 10, 72)
Y_train = Y_train.reshape(-1, 10, 2)
# print('X_train.shape:', X_train.shape, ', Y_train.shape:', Y_train.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(10, 72), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, epochs=10, batch_size=1, validation_split=0.1)

X_test = X_test.reshape(-1, 10, 72)
Y_test = Y_test.reshape(-1, 10, 2)

score = model.evaluate(X_test, Y_test)
print('score;', score)

