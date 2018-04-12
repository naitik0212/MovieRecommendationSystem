import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

import data_reading
import db_helper

filename = 'Data/ml-20m/ratings.csv'
moviesFileName = "Data/ml-20m/movies.csv"

train, test = data_reading.getTrainTestData(filename)

movies_df = pd.DataFrame(data_reading.load_csv_dataset(moviesFileName))


movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
#print(train)

train_ratings_df =  train.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

train_ratings_matrix = train_ratings_df.as_matrix()

rating_mean = np.mean(train_ratings_matrix , axis=1)

train_ratings_matrix_demeaned = train_ratings_matrix - rating_mean.reshape(-1,1)

U, sigma , vt = svds(train_ratings_matrix_demeaned ,  k=50)
sigma = np.diag(sigma)

svd_predicted_ratings = np.dot(np.dot(U, sigma), vt) + rating_mean.reshape(-1, 1)
trained_df = pd.DataFrame(svd_predicted_ratings, columns = train_ratings_df.columns)

#print(train_ratings_df)

def recommendMovies(userid , movies_df , ratings_df , trained_ratings_df , num_of_recommendation = 10):
    user_index = userid - 1
    pred_user = pd.DataFrame(trained_ratings_df.iloc[user_index].sort_values(ascending=False)).reset_index()
    #print(ratings_df)
    user_data = ratings_df[ratings_df.userId == (userid)]
    user_rating_movies_data = user_data.merge(movies_df , how = "left" , left_on = "movieId" , right_on = "movieId").\
        sort_values(['rating'] , ascending=False)

    print("User : " + str(userid) + " has rated " + str(user_rating_movies_data.shape[0]) + " movies. ")

    movies_user_not_rated = movies_df[~movies_df['movieId'].isin(user_rating_movies_data['movieId'])]

    recommends = movies_user_not_rated.merge(pred_user , how= "left" , left_on ="movieId" , right_on = "movieId")

    recommends_sorted = recommends.rename(columns = {user_index: "Preds"}).sort_values('Preds',ascending=False).iloc[:num_of_recommendation,:-1]


    return recommends_sorted , user_rating_movies_data

r , u = recommendMovies(15, movies_df , train , trained_df )

print(r['title'])

#print(u)


#print(train)
#print(test)