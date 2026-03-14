import numpy as py
import pandas as pd

iris = pd.read_csv("iris.csv")

print(iris.isnull().sum())

X = iris[["sepal_length","sepal_width","petal_length","petal_width"]]

y = iris["species"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(xtrain,ytrain)

ytrainp = knn.predict(xtrain)
ytestp = knn.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix
cm_train = confusion_matrix(ytrain, ytrainp)
print(cm_train)
cm_test = confusion_matrix(ytest, ytestp)
print(cm_test)
#calculated the eucledian distance