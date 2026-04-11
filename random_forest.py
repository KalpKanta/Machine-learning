import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

adult = pd.read_csv("adult.csv")

print(adult.isnull().sum())

print(adult.info())

X = adult.iloc[1:, :-1]
y = adult.iloc[1:, -1]

print(X, y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6, random_state = 6)
rfs = RandomForestClassifier()

rfs.fit(xtrain, ytrain)

ytrainp = rfs.predict(xtrain)
ytestp = rfs.predict(xtest)

cm_train = confusion_matrix(ytrain, ytrainp)
cm_test = confusion_matrix(ytest, ytestp)
print(cm_train)
print(cm_test)
print(classification_report(ytrain, ytrainp))
print(classification_report(ytest, ytestp))