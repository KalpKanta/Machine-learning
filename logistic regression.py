import numpy as np
import pandas as pd

titanic = pd.read_csv("titanic.csv")

print(titanic.isnull().sum())

x = titanic[["Pclass", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard"]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x["Sex"] = le.fit_transform(x["Sex"])
print(x)

y = titanic["Survived"]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 6)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtrain, ytrain)

prey = lr.predict(xtrain)
testy = lr.predict(xtest)