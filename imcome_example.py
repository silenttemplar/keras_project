import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '5k']
df = pd.read_csv('./dataset/adult.data', index_col=False, names=names)

def showCSVBrief():
   print(df.head())
   print(df.shape)
   print(df.describe())
   print(df.count())
   print(df.isnull().any())


def showPlotChart():
    # sns.countplot('5k', data=df)
    # sns.countplot('5k', hue='sex', data=df)
    # sns.heatmap(df.corr(), annot=True, cmap='summer_r', linewidths=0.2)
    # sns.violinplot('race', 'age', hue='5k', data=df, split=True)
    # plt.figure(figsize=(10, 10))
    # sns.violinplot('race', 'age', hue='5k', data=df, split=True)
    # plt.show()
    pass

def main():
    Y = df['5k'].values.tolist()
    Y = [1 if i == ' <=50K' else 0 for i in Y]
    Y = to_categorical(Y)
    # print(Y)
    # print(df.head())

    # 숫자로 된 부분을 빼서 문자로 된 것만 얻기
    nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X = df.drop(nums, axis=1)
    X = X.drop('5k', axis=1)
    X = pd.get_dummies(X, drop_first=True)  # 문자열을 숫자로 변경(one hot encoding)
    X = pd.concat([X, df[nums]], axis=1)

    scaler = MinMaxScaler() # normalization (set value in zero to one)
    X[nums] = scaler.fit_transform(X[nums])
    # print(X.head())

    X_train = X[:-1000]
    X_test = X[-1000:]

    Y_train = Y[:-1000]
    Y_test = Y[-1000:]

    Y_train.shape = (31561, 2)
    Y_test.shape = (1000, 2)

    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(100, )))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # print(model.summary())

    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, epochs=5, validation_split=0.2)
    # print(hist.history)

    # plt.figure(figsize=(10, 10))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(hist.history['acc'], color='r')
    # plt.plot(hist.history['val_acc'], color='b')
    # plt.title('acc')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(hist.history['loss'], color='r')
    # plt.plot(hist.history['val_loss'], color='b')
    # plt.title('loss')
    # plt.show()

    score = model.evaluate(X_test, Y_test)
    print(score)

    pred = model.predict(X_test)
    print(pred[:10])
    print(Y[:10])




if __name__ == "__main__":
    # showPlotChart()
    main()

