import numpy as np
import data_reading
from scipy.sparse import coo_matrix
from datetime import datetime
from numpy import sort
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def trainModel(R, K, alpha, beta, epochs):
    numUsers, numItems = R.shape # R is the dense matrix
    u , s , vt = svds(R , k = 30)
    P = u
    s = np.diag(s)
    Q = np.dot(s,vt).transpose()

    # since the values are too large the rmse will take more iterations to converge.
    P = P /10000
    Q = Q / 10000

    userBias = np.zeros(numUsers)  # initialization of user bias
    movieBias = np.zeros(numItems) # initialzation of movie Bias
    globalBias =  np.mean(R[np.nonzero(R)])# initialization of global bias

    print(" Initialization Step completed")
    print("converting to sparse matrix ", str(datetime.now()))
    a = coo_matrix(R)
    a = a.tocsc()
    ratingList = [(i, j, R[i, j]) for i, j in zip(*a.nonzero())]
    print("got desired data at ", str(datetime.now()))

    trainError = []

    for x in range(epochs):
        np.random.shuffle(ratingList)
        P, Q = graidentDescent(ratingList, globalBias, userBias, movieBias, P, Q, alpha, beta)
        error = calculateRootMeanSquareError(globalBias, userBias, movieBias, P, Q)
        trainError.append(error)
        print("Mean Squared Error is %f for iteration %d Completed at %s" % ( error, x+1, str(datetime.now())))
    print(trainError)
    return P , Q , globalBias , userBias , movieBias


def calculateRootMeanSquareError(globalBias, userBias, movieBias, P, Q):
    nonZeroUserIndex, nonZeroMovieIndex = R.nonzero() # getting all the non zero element of matrix.
    error = 0 # initializing error to 0 for every iteration.

    for userIndex, movieIndex in zip(nonZeroUserIndex, nonZeroMovieIndex):
        prediction =  predict(globalBias, userBias, movieBias, userIndex, movieIndex, P, Q)
        error += np.square(R[userIndex, movieIndex] - prediction)

    error /= len(nonZeroUserIndex)

    return np.sqrt(error)

def testError(globalBias, userBias, movieBias, P, Q):
    error = 0
    count = 0 # to find how many ratings are defaulted i.e how many movies user has not rated
    cnt = 0  # Count to find the number od movies user has rated.

    for index, row in test.iterrows():
        uid = row['userId']
        mid = row['movieId']
        if uid in user_index and mid in movie_index:
            u_index = user_index[uid]
            m_index = movie_index[mid]
            prediction = predict(globalBias, userBias, movieBias, u_index, m_index, P, Q)
            cnt += 1
        else:
            count += 1
            prediction = 2.5
        error += np.square(row['rating'] - prediction)

    error /= len(test)

    return np.sqrt(error)


def graidentDescent(ratingList, globalBias, userBias, movieBias, P, Q, alpha, beta):
    for userIndex, movieIndex, rating in ratingList:
        prediction = predict(globalBias, userBias, movieBias, userIndex, movieIndex, P, Q)
        error = rating - prediction

        Pnumerator = error * Q[movieIndex, :] - beta * P[userIndex, :]
        P[userIndex, :] += 2* alpha* Pnumerator/ 10000

        Qnumberator = error * P[userIndex, :] - beta * Q[movieIndex, :]
        Q[movieIndex, :] += 2* alpha* Qnumberator/10000

        userBias[userIndex] += alpha * (error - beta * userBias[userIndex]) # updating the user bias
        movieBias[movieIndex] += alpha * (error - beta * movieBias[movieIndex]) # updating the movie bias

    return P, Q

def predict(globalBias, userBias, movieBias, i, j, P, Q):

    prediction = P[i, :].dot(Q[j, :].T)   + globalBias + userBias[i] + movieBias[j]  # adding all the bias

    return prediction

ratingfilename = 'Data/ml-20m/ratings.csv'
moviesFileName = "Data/ml-20m/movies.csv"

train, test = data_reading.getTrainTestData(ratingfilename)

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
sparse_matrix = sparse_matrix.astype(dtype='float32')
print("sparse ", str(datetime.now()))
R = sparse_matrix.todense()
print("dense ", str(datetime.now()))

P , Q , globalBias , userBias , movieBias= trainModel(R, 30, 0.01, 0.01, 100)

testerror = testError(globalBias , userBias , movieBias , P , Q)

print("training completed at ", str(datetime.now()))
print("Final MSE on test data is " + testerror)
