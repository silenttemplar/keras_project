import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

Data_set = np.loadtxt('./dataset/ThoraricSurgery.csv', delimiter=',')
print('Data_set.shape=', Data_set.shape)

X = Data_set[:, 0:17]
Y = Data_set[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

lst = model.evaluate(X, Y)
print(type(lst), lst)

loss = lst[0]
accuracy = lst[1]

print("\n Loss: %.4f" % loss)  # Loss: 0.1455
print("\n Accuracy: %.4f" % accuracy)  # Accuracy: 0.8511
