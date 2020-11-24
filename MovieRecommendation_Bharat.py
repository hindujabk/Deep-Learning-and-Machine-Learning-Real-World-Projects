#Project 8 - MOvie Reviews using Collaborative Filter Item Based Approach

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the Dataset
movie_titles = pd.read_csv('Movie_Id_Titles')
movie_ratings = pd.read_csv('u.data', sep = '\t', names = ['user_id', 'item_id', 'rating', 'timestamp'])

movie_ratings = movie_ratings.drop(['timestamp'], axis = 1, inplace = True)

movie_ratings.describe()
movie_ratings.info()

movie_rating_df = pd.merge(movie_ratings, movie_titles, on = 'item_id')

#Visualize The Dataset
movie_rating_df.groupby('title').describe()

titles_info = movie_rating_df.groupby('title')['rating'].describe()

titles_mean = movie_rating_df.groupby('title')['rating'].describe()['mean']
titles_count = movie_rating_df.groupby('title')['rating'].describe()['count']

titles = pd.concat([titles_mean, titles_count], axis = 1)

titles['mean'].plot(bins = 100, kind = 'hist', color = 'b')

titles['count'].plot(bins = 50, kind = 'hist', color = 'b')

titles[titles['mean'] == 5]

titles.sort_values('count', ascending = False).head(100)


#Perform Item-Based Collaborative Filtering on One Movie Sample

movie_rating_df

userid_movietitle_matrix =  movie_rating_df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')

titanic = userid_movietitle_matrix['Titanic (1997)']
#titanic = titanic.reset_index()

Starwars = userid_movietitle_matrix['Star Wars (1977)']

titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns = ['Correlation'])
titanic_correlations = titanic_correlations.join(titles['count'])
titanic_correlations.dropna(inplace = True)

titanic_correlations.sort_values('Correlation', ascending = False)

titanic_related = titanic_correlations[titanic_correlations['count'] > 80].sort_values('Correlation', ascending = False)

#Mini Challenge for Starwars correlated movies


starwars_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(Starwars), columns = ['Correlation'])
starwars_correlations = starwars_correlations.join(titles['count'])
starwars_correlations.dropna(inplace = True)

starwars_correlations.sort_values('Correlation', ascending = False)

starwars_related = starwars_correlations[starwars_correlations['count'] > 80].sort_values('Correlation', ascending = False)


#Create an Item Based Collaborative Filter on the Entire Dataset
movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)

myRatings = pd.read_csv('My_Ratings.csv')

similar_movie_list = pd.Series()

for i in range(0,2):
    similar_movie = movie_correlations[ myRatings['Movie Name'][i]  ].dropna()
    similar_movie = similar_movie.map(lambda x : x* myRatings ['Ratings'][i])
    similar_movie_list = similar_movie_list.append(similar_movie)
    
    
similar_movies = similar_movie_list.sort_values(ascending = False)

    