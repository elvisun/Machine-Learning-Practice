from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import keras
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder

def pd2np(data):
	x = data.drop('label', axis = 1).as_matrix()
	y = data[['label']]

	#reshape to 2D
	x = x.reshape(x.shape[0], 1, 28,28)
	return (x, y)

def plotResult(history):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


print("Reading csv...")
rawData = pd.read_csv('train.csv')

train, test = train_test_split(rawData, test_size = 0.2)
	
x_train, y_train = pd2np(train)
x_test, y_test = pd2np(test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# enc = OneHotEncoder()
# enc.fit([0,1,2,3,4,5,6,7,8,9])
# q = enc.transform(y_train).toarray()
# print(q)


model = Sequential()

model.add(Conv2D(512, kernel_size = (3,3), activation='relu', input_shape=(1, 28, 28), padding="same"))
model.add(Conv2D(256, (3,3), activation = 'relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


print("fitting...")
history = model.fit(x_train, y_train, epochs=15, verbose = 1, batch_size = 128, validation_data=(x_test, y_test))

print("cross validating...")
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)

plotResult(history)

# predict
testData = pd.read_csv('test.csv')

ret = model.predict(testData.as_matrix().reshape(testData.shape[0],1,28,28), batch_size = 128)

print(ret)
toSave = np.argmax(ret, axis = 1)
toSave.astype(int)

df = pd.DataFrame(data = toSave, columns = ['label'],index = [i for i in range(1, len(toSave) + 1)])

df['ImageID'] = np.array([i for i in range(1, len(toSave) + 1)])

print(df.head())
df = df[df.columns.tolist()[::-1]]
df.to_csv('cnn-result.csv', sep=',', encoding='utf-8', index=False)

