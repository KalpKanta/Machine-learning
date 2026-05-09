import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
raw_data = load_breast_cancer()
data = pd.DataFrame(raw_data["data"], columns = raw_data["feature_names"])
print(data.head())
#aplying MinMax scaling 
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
scale_data = mms.fit_transform(data)
#aplying pca
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scale_data)
new_data = pca.transform(scale_data)
print(scale_data.shape)
plt.figure(figsize = (10,10))
plt.scatter(new_data[:0], new_data[0:1], c = raw_data["target"])
plt.xlabel("first_principal_component")
plt.ylabel("second_principal_component")
plt.show()