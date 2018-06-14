from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('./dataset/iris.csv',
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# sns.pairplot(df, hue='species')
# plt.show()

dataset = df.values

X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
# print(Y)

Y_encoded = np_utils.to_categorical(Y)
# print(Y_encoded)

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y_encoded, epochs=50, batch_size=1)

print()
print('Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))
