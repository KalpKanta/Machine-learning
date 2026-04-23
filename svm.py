import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

cancer_dictionary = datasets.load_breast_cancer()
cancer_data = pd.DataFrame(cancer_dictionary.data, columns = cancer_dictionary.feature_names)
print(cancer_data)
