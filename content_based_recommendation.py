#content based recommendation
import pandas as pd

movies = pd.read_csv("movies_metadata.csv")

#import tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = "english")
movies["overview"] = movies["overview"].fillna("")
tfidf_matrix = tfidf.fit_transform(movies["overview"])
print(tfidf_matrix.shape)

print(tfidf.get_feature_names_out()[5000:5010])
from sklearn.metrics.pairwise import linear_kernel
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix) # - maps the
indices = pd.Series(movies.index, index = movies["title"]).drop_duplicates() 
def get_recommendation(title, cosine_sim = cosine_similarity):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda X:X[1], reverse = True)
    similarities_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in similarities_scores]
    return movies["title"].iloc[movie_indices]
    
print(get_recommendation("The Dark Knight Rises"))