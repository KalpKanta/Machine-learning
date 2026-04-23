import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

adult = pd.read_csv("adult.csv", sep = ", ")

print(adult.isnull().sum())

le = LabelEncoder()
for column in adult.columns:
    adult[column] = le.fit_transform(adult[column])
"""adult.iloc[:, 3] = le.fit_transform(adult.iloc[:, 3])
adult.iloc[:, 5] = le.fit_transform(adult.iloc[:, 5])
adult.iloc[:, 6] = le.fit_transform(adult.iloc[:, 6])
adult.iloc[:, 7] = le.fit_transform(adult.iloc[:, 7])
adult.iloc[:, 8] = le.fit_transform(adult.iloc[:, 8])
adult.iloc[:, 9] = le.fit_transform(adult.iloc[:, 9])
adult.iloc[:, 13] = le.fit_transform(adult.iloc[:, 13])
adult.iloc[:, 14] = le.fit_transform(adult.iloc[:, 14])"""

print(adult.info())

X = adult.iloc[1:, :-1]
y = adult.iloc[1:, -1]

print(X, y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6, random_state = 6)
dtc = DecisionTreeClassifier()

dtc.fit(xtrain, ytrain)

ytrainp = dtc.predict(xtrain)
ytestp = dtc.predict(xtest)

cm_train = confusion_matrix(ytrain, ytrainp)
cm_test = confusion_matrix(ytest, ytestp)
print(cm_train)
print(cm_test)
print(classification_report(ytrain, ytrainp))
print(classification_report(ytest, ytestp))