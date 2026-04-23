import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

car = pd.read_csv("car.data")
car.columns = ("sales", "maintainance", "doors", "persons", "boot_space", "safety", "class")
    
print(car.isnull().sum())

print(car.info())

le = LabelEncoder()
car["sales"] = le.fit_transform(car["sales"])
car["maintainance"] = le.fit_transform(car["maintainance"])
car["boot_space"] = le.fit_transform(car["boot_space"])
car["safety"] = le.fit_transform(car["safety"])
car["class"] = le.fit_transform(car["class"])
car["doors"] = le.fit_transform(car["doors"])
car["persons"] = le.fit_transform(car["persons"])
print(car.info())
print(car.corr())

X = car[["sales", "maintainance", "boot_space", "safety", "doors", "persons"]]
y = car["class"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6, random_state = 6)
from sklearn.tree import DecisionTreeClassifier

dtn = DecisionTreeClassifier()
dtn.fit(xtrain, ytrain)

ytrainp = dtn.predict(xtrain)
ytestp = dtn.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix
cm_train = confusion_matrix(ytrain, ytrainp)
print(cm_train)
cm_test = confusion_matrix(ytest, ytestp)
print(cm_test)
print(classification_report(ytrain,ytrainp))
print(classification_report(ytest,ytestp))