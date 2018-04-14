import pandas as pd
import numpy as np
import data_reading
import db_helper
from sklearn.utils.extmath import randomized_svd
import copy

class LatentFactorModel(object):
	def processData(self, file1, file2):
		train, test = data_reading.getTrainTestData(filename)
		movies_df = pd.DataFrame(data_reading.load_csv_dataset(moviesFileName))
		movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
		train_ratings_df =  train.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
		train_ratings_df_T = train_ratings_df.T
		train_ratings_matrix = train_ratings_df_T.as_matrix()

		user_ids = train_ratings_df.index.values.tolist() 
		movie_ids = list(train_ratings_df)

		return train, test, train_ratings_matrix, user_ids, movie_ids

	def SVDFactorization(self, train_ratings_matrix):
		U, sigma , vt = randomized_svd(train_ratings_matrix ,  n_components=50)
		sigma = np.diag(sigma)
		Q = U
		P = np.dot(sigma, vt)

		return P, Q

	def calculateRMSE(self, P, Q):
	    total = 0
	    for index, row in train.iterrows():
	        rxi = row['rating']
	        user_id = row['userId']
	        movie_id = row['movieId']
	        user_index = user_ids.index(user_id)
	        movie_index = movie_ids.index(movie_id)
	        qi = Q[movie_index]
	        pxt = P[:,user_index]
	        predicted_rating = np.dot(qi, pxt)
	        
	        total += (rxi - predicted_rating) ** 2
	    
	    return np.sqrt(total)

	def gradiantDescent(self, P, Q, user_ids, movie_ids, train):
	    print ("GD")
	    rmse = self.calculateRMSE(P, Q)
	    for index in range(4):
	        print(index, rmse)
	        eta = 0.01
	        l = 0.5
	        tmp = (len(Q), len(Q[0]))
	        delta_Q = np.zeros(tmp)
	        tmp = (len(P), len(P[0]))
	        delta_P = np.zeros(tmp)
	        
	        for index, row in train.iterrows():
	            user_id = row['userId']
	            movie_id = row['movieId']
	            #print(row['rating'])
	            user_index = user_ids.index(user_id)
	            movie_index = movie_ids.index(movie_id)
	            #print (user_id, movie_id, user_index, movie_index)
	            
	            qi = Q[movie_index]
	            pxt = P[:,user_index]
	            #print (pxt)
	            predicted_rating = np.dot(qi, pxt)
	            #print (predicted_rating)
	            
	            diff_rating = -2 * (row['rating'] - predicted_rating)
	            for i in range(len(Q[0])):
	                pxk = pxt[i]
	                qik = qi[i]
	                delta_Q[movie_index][i] = eta * ((diff_rating * pxk) + 2*l*qik)
	                delta_P[i][user_index] = eta * ((diff_rating * qik) + 2*l*pxk)
	            
	            new_Q = Q - delta_Q
	            new_P = P - delta_P
	            
	            P = copy.deepcopy(new_P)
	            Q = copy.deepcopy(new_Q)
	        
	            rmse = self.calculateRMSE(P, Q)


if __name__=="__main__":
	filename = 'Data/ml-20m/ratings.csv'
	moviesFileName = "Data/ml-20m/movies.csv"

	lfm  = LatentFactorModel()

	train, test, train_ratings_matrix, user_ids, movie_ids = lfm.processData(filename, moviesFileName)
	P, Q = lfm.SVDFactorization(train_ratings_matrix)
	lfm.gradiantDescent(P, Q, user_ids, movie_ids, train)


