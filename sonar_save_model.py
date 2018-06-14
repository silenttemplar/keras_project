from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import os

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = MODEL_DIR+'my_model.hdf5'

df = pd.read_csv('./dataset/sonar.csv', header=None)
dataset = df.values
X = dataset[:, 0:-1]
Y_obj = dataset[:, -1]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=130, batch_size=5)


model.save(modelpath)  # 모델을 컴퓨터에 저장. >> h5py package 필요

del model       # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model(modelpath) # 모델을 새로 불러옴

