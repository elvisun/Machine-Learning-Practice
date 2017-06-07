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



def convertSex(x):
	if x == 'male':
		return 1
	return 0

def parseCabin(x, s):
	if not (pd.isnull(x)):
		if not s:
			return x[0]
		else:
			return x[1:]
	else:
		return np.nan

def parseTicket(x, s):
	res = x.split()
	if s == 0:
		if len(res) > 1:
			return res[0]
		else:
			return np.nan
	else:
		return res[-1]

def fillNan(x, average):
	if np.isnan(x):
		return average
	else:
		return x

def preProcessing(rawData):
	rawData['Sex'] = rawData['Sex'].apply(convertSex)
	rawData['CabinS'] = rawData['Cabin'].apply(parseCabin, args=(0,))
	rawData['CabinN'] = rawData['Cabin'].apply(parseCabin, args=(1,))
	rawData['TicketS'] = rawData['Ticket'].apply(parseTicket, args=(0,))
	rawData['TicketN'] = rawData['Ticket'].apply(parseTicket, args=(1,))

	rawData['Age'] = rawData['Age'].apply(fillNan, args=(float(int(rawData['Age'].median())), ))
	rawData['Fare'] = rawData['Fare'].apply(fillNan, args=(float(int(rawData['Fare'].mean())), ))

	labelColumns = ['Embarked', 'CabinS', 'TicketS']
	for label in labelColumns:
		le = preprocessing.LabelEncoder()
		le.fit(rawData[label])
		leNameMapping = dict(zip(le.classes_, le.transform(le.classes_)))
		rawData[label] = rawData[label].apply(lambda x: leNameMapping[x])

	return rawData


def main():

	rawData = pd.read_csv('train.csv')

	# fig = plt.figure(figsize = (10,10))
	# fig_dims = (3,2)
	# plt.subplot2grid(fig_dims, (0,0))

	rawData['Survived'].value_counts().plot(kind = 'bar', title='Death and Survival')
	#plt.show()
	rawData = preProcessing(rawData)
	#print (rawData['Embarked'].unique())

	#print(rawData.head(20))
	#count nan
	
	#split into training and cross validating
	train, test = train_test_split(rawData, test_size = 0.2)

	
	nnModel = MLPClassifier(max_iter = 20000)
	parameters = {
	'max_iter': [1000, 10000],
	'activation' :['relu', 'tanh', 'logistic'],
	'learning_rate_init': [0.001],
	}
	clf = GridSearchCV(nnModel, parameters)
	clf.fit(train.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked','CabinS','TicketS']].as_matrix(), train.Survived.values)
	print clf.cv_results_['mean_test_score']
	print clf.best_params_

	# lgModel.fit(train.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked','CabinS','TicketS']].as_matrix(), train.Survived.values)
	# lgscores = cross_val_score(lgModel, test.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare']].as_matrix(), test.Survived.values, cv=5)
	# print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (lgscores.mean(), lgscores.std() * 2))
	
	

	# rfModel = RandomForestClassifier(n_estimators=100)
	# rfModel.fit(train.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked','CabinS','TicketS']].as_matrix(), train.Survived.values)
	# rfscores = cross_val_score(rfModel, test.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare']].as_matrix(), test.Survived.values, cv=5)
	# print("Random Forest Regression Accuracy: %0.2f (+/- %0.2f)" % (rfscores.mean(), rfscores.std() * 2))
	

	# nnModel = MLPClassifier(max_iter = 20000)
	# nnModel.fit(train.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked','CabinS','TicketS']].as_matrix(), train.Survived.values)
	# nnscores = cross_val_score(nnModel, test.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare']].as_matrix(), test.Survived.values, cv=5)
	# print("Neural Network Accuracy: %0.2f (+/- %0.2f)" % (nnscores.mean(), nnscores.std() * 2))
	

	finalModel = clf

	testData = preProcessing(pd.read_csv('test.csv'))
	#print(testData.isnull().sum())
	df = pd.DataFrame(finalModel.predict(testData.loc[:, ['Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked','CabinS','TicketS']].as_matrix()))
	df['PassengerId'] = testData['PassengerId']
	df.columns = ['Survived', 'PassengerId']
	df = df[df.columns.tolist()[::-1]]	#flip column
	#print df
	#print(testData)
	df.to_csv('result.csv', sep=',', encoding='utf-8', header=["PassengerId", "Survived"],  index=False)

if __name__ == '__main__':
	main()