#multi-variable : y = m1,x1 + m2,x2 + m3,x3 ...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv("iris.csv")
print(iris.info())
x = iris.iloc[:,:-1]
y = iris.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#splitting data into training and testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.6, random_state = 6)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(xtrain, ytrain)
m = lr.coef_
c = lr.intercept_
print(m)
print(c)

prey = lr.predict(xtrain)
testy = lr.predict(xtest)

from sklearn.metrics import root_mean_squared_error
error1 = root_mean_squared_error(ytrain, prey)
error2 = root_mean_squared_error(ytest, testy)
print(error1)
print(error2)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pf = PolynomialFeatures(degree = 2)
xpf = pf.fit_transform(xtrain)

lr = LinearRegression()
lr.fit(xpf, ytrain)
m = lr.coef_
xtest_poly = pf.transform(xtest)

y = lr.intercept_
prey = lr.predict(xtest_poly)

error1 = root_mean_squared_error(ytest, prey)
print(error1)