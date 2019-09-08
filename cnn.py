import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from numpy import ndarray
from keras.models import load_model
from sklearn.utils import Bunch

# 定義梯度下降批量
batch_size = 128
# 定義分類數量
num_classes = 10
# 定義訓練週期
epochs = 1
# 定義圖像寬、高
img_rows, img_cols = 28, 28

# print(K.image_data_format())


def fetch_data():
    with open('D:/documents/ML_ch03/dataobj', 'rb') as file:
        mnist: Bunch = pickle.load(file)
    return mnist


# Dictionary-like object
mnist: Bunch = fetch_data()
X: ndarray = mnist["data"]  # (70000, 784)
y: ndarray = mnist["target"]  # (70000,)

# shuffle
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index: ndarray = np.random.permutation(60000)  # generate a 0-59999 random permutation
# break the data order
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# channels_first: 色彩通道(R/G/B)資料(深度)放在第2維度，第3、4維度放置寬與高
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:  # channels_last: 色彩通道(R/G/B)資料(深度)放在第4維度，第2、3維度放置寬與高
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 轉換色彩 0~255 to 0~1 float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
# y 值轉成 one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filters=32,即 即 output depth, Kernal Size: 3x3, activation function = relu
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 建立卷積層，filters=64,即 output depth, Kernal Size: 3x3, activation function = relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# 建立池化層，池化大小=2x2  max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合 25%
model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(units=128, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合 50%
model.add(Dropout(0.5))
# 使用 softmax activation function，將結果分類 10 outputs
model.add(Dense(units=num_classes, activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test))

# 顯示損失函數、訓練成果(分數)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model
model.save('mnist_cnn_model.h5')

# load model
model = load_model('mnist_cnn_model.h5')
