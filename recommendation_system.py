import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

movie = pd.read_csv("movies_metadata.csv")
print(movie.head())
print(movie.info())
#weiogtted rating
(v/v+m)*r+(m/v+m)*c
#v = number of votes
#m = minimum votes required to be listed in chart 
#r = average rating for movie
#c = mean across the whole report 
c = movie["vote_average"].mean()
m = movie["vote_count"].quantile(0.90)
q_movies =  movie.copy().loc[movie["vote_count"]>=m]
print(q_movies.shape)
def wighted_rating(x,m=m,c=c):
    pass

