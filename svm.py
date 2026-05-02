import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn import metrics

cancer_dictionary = datasets.load_breast_cancer()
cancer_data = pd.DataFrame(cancer_dictionary.data, columns = cancer_dictionary.feature_names)
cancer_data["iscancer"] = cancer_dictionary.target
print(cancer_data)
print(cancer_data.info())

y = cancer_data["iscancer"]
cancer_data.drop("iscancer", axis = 1)
X = cancer_data

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6, test_size = 0.4, random_state = 6)
from sklearn import svm
cls = svm.SVC(kernel = "linear")
cls.fit(xtrain, ytrain)
ytestp = cls.predict(xtest)
ytrainp = cls.predict(xtrain)
print("accuracy_score", metrics.accuracy_score(ytest, ytestp))

cm_train = confusion_matrix(ytrain, ytrainp)
print(cm_train)
cm_test = confusion_matrix(ytest, ytestp)
print(cm_test)
