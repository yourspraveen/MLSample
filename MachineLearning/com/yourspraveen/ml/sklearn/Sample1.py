'''
Created on Jul 8, 2017

@author: Praveen Palaniswamy
'''

from sklearn import datasets

#getting data set of Iris
iris = datasets.load_iris()

#features -> Input
X = iris.data

#expected results -> Output
y = iris.target

#generating Training and Test data sets as equal split of iris data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#classifier algorithm -> Decision Tree
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier()

#classifier algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

#Training
classifier.fit(X_train, y_train)

#Test Data set for verification
predictions = classifier.predict(X_test)

#Result set of test data set
#print(dTreePredictions)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
