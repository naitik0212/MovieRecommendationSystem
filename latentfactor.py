import numpy as np
import data_reading
import pandas as pd
from scipy.sparse import coo_matrix
import os
from datetime import datetime
from numpy import sort
from scipy.sparse import csr_matrix


def trainModel(R, K, alpha, beta, iterations):
    # Initialize user and item latent feature matrice
    num_users, num_items = R.shape
    P = np.random.normal(loc=0, scale=1. / (K), size=(num_users, K))
    Q = np.random.normal(loc=0, scale=1. / (K), size=(num_items, K))

    # Initialize the biases
    userBias = np.zeros(num_users)
    movieBias = np.zeros(num_items)
    b = np.mean(R[np.where(R != 0)])

    print("converting to sparse matrix ", str(datetime.now()))
    a = coo_matrix(R)
    a = a.tocsc()
    ratingList = [(i, j, R[i, j]) for i, j in zip(*a.nonzero())]
    print("got desired data at ", str(datetime.now()))

    # Perform stochastic gradient descent for number of iterations
    for i in range(iterations):
        np.random.shuffle(ratingList)
        stochasticGraidentDescent(ratingList, b, userBias, movieBias, P, Q, alpha, beta)
        error = testError(b, userBias, movieBias, P, Q)
        print("Iteration: %d ; error = %.4f" % (i + 1, error))

    return

def calculateRootMeanSquareError(b, userBias, movieBias, P, Q):
    nonZeroUserIndex, nonZeroMovieIndex = R.nonzero()
    error = 0
    for x, y in zip(nonZeroUserIndex, nonZeroMovieIndex):
        error += pow(R[x, y] - get_rating(b, userBias, movieBias, x, y, P, Q), 2)
    error /= len(nonZeroUserIndex)
    return np.sqrt(error)

def testError(b, userBias, movieBias, P, Q):
    error = 0
    count = 0
    cnt = 0
    for index, row in test.iterrows():
        uid = row['userId']
        mid = row['movieId']
        predicted_rating = 0

        if uid in user_index and mid in movie_index:
            u_index = user_index[uid]
            m_index = movie_index[mid]
            predicted_rating = get_rating(b, userBias, movieBias, u_index, m_index, P, Q)
            cnt += 1
        else:
            count += 1
            predicted_rating = 2.5
        error += np.square(row['rating'] - predicted_rating)
    print(cnt, count)
    error /= len(test)

    return np.sqrt(error)

def stochasticGraidentDescent(ratingList, b, userBias, movieBias, P, Q, alpha, beta):
    for i, j, r in ratingList:
        # Computer prediction and error
        prediction = get_rating(b, userBias, movieBias, i, j, P, Q)
        e = (r - prediction)

        # Update biases
        userBias[i] += alpha * (e - beta * userBias[i])
        movieBias[j] += alpha * (e - beta * movieBias[j])

        # Update user and item latent feature matrices
        temp_p = alpha * (e * Q[j, :] - beta * P[i, :]) / len(ratingList)
        P[i, :] += temp_p

        temp_q = alpha * (e * P[i, :] - beta * Q[j, :]) / len(ratingList)
        Q[j, :] += temp_q

def get_rating(b, userBias, movieBias, i, j, P, Q):
    prediction = b + userBias[i] + movieBias[j] + P[i, :].dot(Q[j, :].T)
    return prediction

filename = 'Data/ml-20m/ratings.csv'
moviesFileName = "Data/ml-20m/movies.csv"

train, test = data_reading.getTrainTestData(filename)
train = train.drop('timestamp', 1)
test = test.drop('timestamp', 1)

user_u = list(sort(train.userId.unique()))
movie_u = list(sort(train.movieId.unique()))

movie_index = {}
for i in range(len(movie_u)):
    movie_index[movie_u[i]] = i
print("movie index dict created")
user_index = {}
for i in range(len(user_u)):
    user_index[user_u[i]] = i

print("User index dict created")

data = train['rating'].tolist()
row = train.userId.astype('category', categories=user_u).cat.codes
col = train.movieId.astype('category', categories=movie_u).cat.codes
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(movie_u)))
print("sparse ", str(datetime.now()))
R = sparse_matrix.todense()
print("dense ", str(datetime.now()))

trainModel(R, 30, 0.1, 0.01, 100)
print("training completed at ", str(datetime.now()))