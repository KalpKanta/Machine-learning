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

#try 