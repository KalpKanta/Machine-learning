import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.txt", delimiter = ";", names = ["text", "label"])
print(train.head())
print(train["label"].value_counts())

def custom_encoder(data):
    data.replace(to_replace = "surprise", value = 1, inplace = True)
    data.replace(to_replace = "love", value = 1, inplace = True)
    data.replace(to_replace = "fear", value = 0, inplace = True)
    data.replace(to_replace = "anger", value = 0, inplace = True)
    data.replace(to_replace = "sadness", value = 0, inplace = True)
    data.replace(to_replace = "joy", value = 1, inplace = True)

custom_encoder(train["label"])

import re 
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()
def text_transformation(data):
    corpus = []
    for sentence in data:
        new_item = re.sub("[^a-zA-Z]", " ", str())
        new_item = new_item.lower()
        new_item = new_item.split()