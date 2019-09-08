
import pickle
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from keras.models import load_model


def fetch_data():
    with open('D:/documents/ML_ch03/dataobj', 'rb') as file:
        mnist: Bunch = pickle.load(file)
    return mnist


mnist: Bunch = fetch_data()
# Dictionary-like object
print(mnist)
X: ndarray = mnist["data"]  # (70000, 784)
y: ndarray = mnist["target"]  # (70000,)

print(X.shape)
print(y.shape)

plt.imshow(X[0].reshape(28, 28))
# print(X[0].reshape(28, 28))
plt.show()  # 5

# shuffle
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index: ndarray = np.random.permutation(60000)  # generate a 0-59999 random permutation
# break the data order
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# normalize
X_norm = X_train/255.0
X_test_norm = X_test/255.0

# 將 training 的 label 進行 one-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# 建立簡單的線性執行的模型
model = Sequential()
# Add Input layer 784 inputs, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
# Add output layer 有 10個輸出變數
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history中
train_history = model.fit(x=X_train, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

# 顯示訓練成果
scores = model.evaluate(X_test_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

# 預測
X = X_test_norm[0:10, :]
predictions = model.predict_classes(X)
predictions_prob = model.predict_proba(X)
# get prediction result
print(predictions, predictions_prob)

# save model
model.save('mnist_model.h5')

# load model
model = load_model('mnist_model.h5')
