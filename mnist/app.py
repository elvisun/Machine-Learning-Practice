#start here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def main():

	rawData = pd.read_csv('train.csv')
	rawData = rawData.head(1000)
	
	#split into training and cross validating
	train, test = train_test_split(rawData, test_size = 0.2)
	
	knn = KNeighborsClassifier(n_neighbors = 5)
	knn.fit(train.drop('label', axis = 1).as_matrix(), train.label.values)
	knnScore = cross_val_score(knn, test.drop('label', axis = 1).as_matrix(), test.label.values, cv=5)
	print("KNN Accuracy: %0.2f (+/- %0.2f)" % (knnScore.mean(), knnScore.std() * 2))

	finalModel = knn

	testData = pd.read_csv('test.csv')

	
	#print(testData.head())
	#print(testData.isnull().sum())
	print (testData.head())
	df = pd.DataFrame(finalModel.predict(testData.as_matrix()))
	
	testData['ImageID'] = range(1, len(testData) + 1)
	df['ImageID'] = testData['ImageID']
	df.columns = ['ImageID', 'label']
	df = df[df.columns.tolist()[::-1]]
	#print df
	#print(testData)
	df.to_csv('result.csv', sep=',', encoding='utf-8', header=["ImageID", "label"],  index=False)

if __name__ == '__main__':
	main()