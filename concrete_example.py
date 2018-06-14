import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# xlrd (Install xlrd >= 0.9.0 for Excel support)
import pandas as pd

from keras.layers import *
from keras.models import *
from keras.utils import *

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split


df1 = pd.read_excel('./dataset/Concrete_Data.xls')
# print(df1.head())
# print(df1.describe())
# print(df1.columns)
df = df1.rename(columns={
    'Cement (component 1)(kg in a m^3 mixture)':'cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'blast',
    'Fly Ash (component 3)(kg in a m^3 mixture)':'fly',
    'Water  (component 4)(kg in a m^3 mixture)':'water',
    'Superplasticizer (component 5)(kg in a m^3 mixture)':'super',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coaese',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine',
    'Age (day)':'age',
    'Concrete compressive strength(MPa, megapascals) ':'strength'
})

X1 = df.drop(['strength'], axis=1)
# print(X1.head())

Y = df['strength']
# print(Y.head())

scaler = MinMaxScaler()
X = scaler.fit_transform(X1)
# print(X.shape)

# sns.pairplot(df)
# plt.show()

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(8,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
# print(model.summary())

model.compile(loss='mse', optimizer='adam')
# model.compile(loss='mse', optimizer='sgd')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# print(X_train.shape)

batch_size = 400
n_iters = 3000
data_length = len(X_train)
batchs = ( data_length / batch_size )
epochs = int( n_iters / batchs )

hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

score = model.evaluate(X_test, Y_test)
print(score)

pred = model.predict(X_test[-5:])
print(pred)
print(Y_test[-5:])

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('loss')
plt.show()