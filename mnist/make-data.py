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

def graphDigit(data):
	print(data)
	label = data[['label']].values[0][0]
	print(label)
	pixels = data.drop('label', axis = 1).values
	pixels = pixels.reshape((28,28))
	plt.title('Label is {label}'.format(label=label))
	plt.imshow(pixels, cmap='gray')
	plt.show()

def synthesisData(df):
	l = df.shape[0]	#get length
	for j in range(0, l):
		print('{} of {} examples'.format(j, l))
		oldMatrix = df.iloc[[j]].drop('label', axis = 1).values.reshape(28,28)
		newMatrices = []
		for _ in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
			newPixels = np.roll(oldMatrix, _[0], axis = 0)
			newPixels = np.roll(newPixels, _[1], axis = 1).reshape(1, 28*28)
			newDF = pd.DataFrame(newPixels, columns = [i for i in range(1, 28*28 + 1)])
			newDF['label'] = df.iloc[[j]]['label']
			#print(newDF)
			df.append(newDF)
			#graphDigit(newDF)
	return df

print("Reading csv...")
rawData = pd.read_csv('train.csv')
rawData = synthesisData(rawData)
df.to_csv('extended-data.csv', sep=',', encoding='utf-8', index=False)