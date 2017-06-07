#start here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


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

def clean_data(df, drop_passenger_id):
    
    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())
    
    # Generate a mapping of Sex from a string to a number representation    
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    
    # Get the unique values of Embarked
    embarked_locs = sorted(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation        
    embarked_locs_mapping = dict(zip(embarked_locs, 
                                     range(0, len(embarked_locs) + 1)))
    
    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)
    
    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3, 
    # we assign the missing values in Embarked to 'S':
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val' : 
                       { embarked_locs_mapping[np.nan] : embarked_locs_mapping['S'] 
                       }
                   }, 
                   inplace=True)
    
    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.  
    # We'll use the median instead of the mean because the Age 
    # histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
                        .groupby([df['Sex_Val'], df['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
            
    # Define a new feature FamilySize that is the sum of 
    # Parch (number of parents or children on board) and 
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    
    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)
    
    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    
    return df


def main():

	clf = RandomForestClassifier(n_estimators=100)

	train_data = clean_data(pd.read_csv('train.csv'), drop_passenger_id=True).values
	# Training data features, skip the first column 'Survived'
	train_features = train_data[:, 1:]

	# 'Survived' column values
	train_target = train_data[:, 0]

	# Fit the model to our training data
	clf = clf.fit(train_features, train_target)
	score = clf.score(train_features, train_target)

	df_test = pd.read_csv('test.csv')
	df_test.head()

	# Data wrangle the test set and convert it to a numpy array
	df_test = clean_data(df_test, drop_passenger_id=False)
	test_data = df_test.values

	# Get the test data features, skipping the first column 'PassengerId'
	test_x = test_data[:, 1:]

	# Predict the Survival values for the test data
	test_y = clf.predict(test_x)


	df_test['Survived'] = test_y
	df_test[['PassengerId', 'Survived']] \
	    .to_csv('results-rf.csv', index=False)


if __name__ == '__main__':
	main()