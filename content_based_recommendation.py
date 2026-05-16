#content based recommendation
import pandas as pd

movies = pd.read_csv("movies_metadata.csv")

#import tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = "English")
movies["overview"] = movies["overview"].fillna("")
tfid_matrix = tfidf.fit_transform(movies["overview"])
print(tfid_matrix.shape)