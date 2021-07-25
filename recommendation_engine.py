# Importing the required Libraries
import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.neighbors import NearestNeighbors
import random

# Let's create a class for the regular functions the will make up the movie recommender system

class Recommendation_tools:
    @staticmethod
    def get_user_preference():
        # reading ratings file:
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

        # reading items file:
        i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
        'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')


        id_ =  items['movie id'].to_list()
        movie_name  = items['movie title'].to_list()

        movie_id = {x:y for x,y in zip(id_ , movie_name)}

        ratings['movie_title'] = ratings.movie_id.replace(movie_id)

        user_preference =  ratings[['user_id','movie_title','rating']]
        return user_preference
    
    @staticmethod
    def recommend_users(user_preference,user_id =1,return_n_users  = 5,metric_to_use = 'cosine'):
        #Generates a matrix of movies title and userid with the corresponding rating as the values
        data_matrix = user_preference.pivot_table(values = 'rating',columns ='movie_title',index ='user_id').fillna(0)
        # Generates a sparse matrix
        sparse_data_matrix = csr_matrix(data_matrix)
        # knn model for selecting similary users
        user_knn_model = NearestNeighbors(n_neighbors=10,metric=metric_to_use, algorithm='brute', n_jobs=-1)
        user_knn_model.fit(sparse_data_matrix)

        # index of target user
        query_index = user_id - 1
        # Generate a sparse query
        sparese_query =  csr_matrix(data_matrix.iloc[query_index,:].values.reshape(1, -1))
        # predict n similar users 
        distances, indices = user_knn_model.kneighbors(sparese_query,n_neighbors=return_n_users+1)

        # a list of user_id  that are similar to target_user id
        top_n_users = indices[0][1:]
        # return a data matrix of movies title and userid with the corresponding rating as the values for only user_id that is similar to the target
        return data_matrix.iloc[top_n_users]
    
    @staticmethod
    def recommmend_movies(similar_users_data_matrix,movie_title = 'Legends of the Fall (1994)',return_n_movies  =5,metric_to_use = 'cosine'):
        #Generates a  Transposed matrix of movies title and userid with the corresponding rating as the values for similar users
        movie_data_matrix = similar_users_data_matrix.T 
        # Generates a sparse matrix
        sparse_movie_data_matrix = csr_matrix(movie_data_matrix)
        # Generates a dictionary of movie to index 
        unique_movie_list = movie_data_matrix.index
        movie_title_index = {unique_movie_list[index]:index  for index in range(len(unique_movie_list))}
        # knn model for selecting similary movies
        knn_model = NearestNeighbors(n_neighbors=10,metric=metric_to_use, algorithm='brute', n_jobs=-1)
        knn_model.fit(sparse_movie_data_matrix)

        # index of target movie
        query_index = movie_title_index[movie_title]
        # Generate a sparse query
        sparese_query_movie =  csr_matrix(movie_data_matrix.iloc[query_index,:].values.reshape(1, -1))
        # predict similar n movies
        distances, indices = knn_model.kneighbors(sparese_query_movie,n_neighbors=return_n_movies+1)

        # list of all similar movie names
        top_n_movies = [movie_data_matrix.index[indices.flatten()][movie] for movie in range(1,len(indices[0]))]
        top_n_movies_distance =  distances[0][1:]
        top_movies_and_distances_from_target  = pd.DataFrame(np.array([top_n_movies,top_n_movies_distance]).T,columns  = ['top_n_movies',metric_to_use +'_distance_top_n_movies'])


        return top_movies_and_distances_from_target

    @staticmethod
    def highly_rated_movies(user_preference):
        # Generates a count of movies reviews per movie
        movie_by_rating = user_preference.groupby('movie_title',).rating.count()
        movie_by_rating = movie_by_rating.reset_index()
        movie_by_rating.columns = ['movie_title','no_rating_recieved_by_movie']
        # Return movies with more than 150 ratings
        top_movies = movie_by_rating[movie_by_rating.no_rating_recieved_by_movie > 150].movie_title.to_list()
        return top_movies
    
    @staticmethod
    def available_metrics():
        print('list of available metrics: \n')
        for metric_ in sklearn.neighbors.VALID_METRICS['brute']:
            print(metric_)
         
        
################# let's create a class for recommending movies   ###################
###################################################################################$

class Movie_recomendation_system(Recommendation_tools):
    @staticmethod
    def recommendation_by_favourite_movie(target_user,fav_movie,metric_to_use = 'cosine'):
        """
        a method that recommends movies to a user based on the user's favorite movie 
        
        """
        '''# Enter the target user ID
        target_user = int(input('Enter the target_id of the user: '))
        fav_movie = input('Enter the favourite movie of the user: ')'''
        
        #  load the user_preference data containing the user_id,movie_title and ratings
        user_preference = Recommendation_tools.get_user_preference()
        
        # Generates a list of movies with high rating counts
        top_movies_with_reviews =  Recommendation_tools.highly_rated_movies(user_preference)

        # Generates an array of interesting movies previously watched by the target user i.e movies the target user gave a rating of 5
        movies_watched_by_target_user =  user_preference[(user_preference.user_id == target_user) & (user_preference.rating >=5)].movie_title.unique()
        # Generate similar users to target users
        similar_users_preference = Recommendation_tools.recommend_users(user_preference = user_preference,user_id=target_user,metric_to_use=metric_to_use)

        # similar movies to nth target movie
        movies_for_selected_user =  Recommendation_tools.recommmend_movies(similar_users_preference,movie_title=fav_movie,metric_to_use=metric_to_use)
        # Generate a set of the generated similar movies 
        total_unique_movies_recommended = movies_for_selected_user.top_n_movies.unique()
        # Generate a list of recommended movies that have been watched by the target user
        true_recommended_movies =  [movie for movie in total_unique_movies_recommended if movie not in movies_watched_by_target_user]
        
        return true_recommended_movies
    
    @staticmethod
    def automatic_recommendation(target_user,metric_to_use = 'cosine'):
        
        """
        a method that recommends movies automatically to a user based on the popular movies the user has watched in the past
        
        """
        '''# Enter the target user ID
        target_user = int(input('Enter the target_id of the user: '))'''
        
         #  load the user_preference data containing the user_id,movie_title and ratings
        user_preference = Recommendation_tools.get_user_preference()
        
        # Generates a list of movies with high rating counts
        top_movies_with_reviews =  Recommendation_tools.highly_rated_movies(user_preference)

        # Generates an array of interesting movies previously watched by the target user i.e movies the target user gave a rating of 5
        movies_watched_by_target_user =  user_preference[(user_preference.user_id == target_user) & (user_preference.rating >=5)].movie_title.unique()
        # Set random seed
        #random.seed(random.sample([99,1,101,23,42],1)[0])
        # get 3 random movies liked by the user
        movies_watched_by_target_user = random.sample(list(movies_watched_by_target_user),3)
        # Generate similar users to target users
        similar_users_preference = Recommendation_tools.recommend_users(user_preference = user_preference,user_id=target_user,metric_to_use=metric_to_use)


        # A DataFrame for storing all similar recommended movies
        total_movies_recommended_for_target_user = pd.DataFrame()

        #Find similar movies to each movie that the target user has watched before 
        for movie in movies_watched_by_target_user:
            # similar movies to nth target movie
            movies_for_selected_user =  Recommendation_tools.recommmend_movies(similar_users_preference,movie_title=movie,metric_to_use=metric_to_use)
            # append nth similar movie to total movies recommended for the target user
            total_movies_recommended_for_target_user = pd.concat([total_movies_recommended_for_target_user,movies_for_selected_user])
        # Generate a set of the generated total similar movies 
        total_unique_movies_recommended = total_movies_recommended_for_target_user.top_n_movies.unique()
        # Generate a list of recommended movies that have been watched by the target user
        true_recommended_movies =  [movie for movie in total_unique_movies_recommended if movie not in movies_watched_by_target_user]
        # Generate a list of recommended movies that is among the highly rated movies
        final_movies_recomendation = [movie for movie in true_recommended_movies if movie in top_movies_with_reviews]
        # return highly rated recommended movies
        return final_movies_recomendation
