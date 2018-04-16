import numpy as np
import data_reading
import pandas as pd
from scipy.sparse import coo_matrix
import os
from datetime import datetime

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        print("training started at ", str(datetime.now()))
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        print("converting to sparse matrix at ", str(datetime.now()))
        self.R = coo_matrix(self.R)
        print("converted to sparse matrix at ", str(datetime.now()))
        self.R = self.R.tocsc()
        print("converted to csc matrix at ", str(datetime.now()))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            if (i + 1) % 10 == 0:
                print("iteration %d at %s" % (i, str(datetime.now())))
            # np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 1 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j in zip(*self.R.nonzero()):
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (self.R[i,j] - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)



def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def compressDataframe(dataframe):
    gl_int = dataframe.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')

    gl_float = dataframe.select_dtypes(include=['float'])
    converted_float = gl_float.apply(pd.to_numeric, downcast='float')

    optimized_gl = dataframe.copy()
    optimized_gl[converted_int.columns] = converted_int
    optimized_gl[converted_float.columns] = converted_float

    return optimized_gl



# hdf = pd.HDFStore('storage.h5')
# hdf.put('d1', train, format='table', data_columns=True)
# print hdf['d1'].shape
# hdf.close()

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

filename = 'Data/ml-20m/ratings.csv'
moviesFileName = "Data/ml-20m/movies.csv"

print("going to load data ", str(datetime.now()))
if os.path.exists("Data/usermoviematrix.npy"):
    print("file already exists")
    R = np.load("Data/usermoviematrix.npy")
else:
    print("file does not exist")
    train, test = data_reading.getTrainTestData(filename)
    train = train.drop('timestamp', 1)
    test = test.drop('timestamp', 1)

    optimized_train = compressDataframe(train)
    train.info(memory_usage='deep')
    print("\n\n\n\n")
    optimized_train.info(memory_usage='deep')
    train_ratings_df = optimized_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R = train_ratings_df.as_matrix()
    np.save("Data/usermoviematrix.npy", R)

print("data loaded at ", str(datetime.now()))
# R = coo_matrix(R)
# print("converted to sparse matrix at ", str(datetime.now()))
# R = R.tocsc()
# print("converted to csc matrix at ", str(datetime.now()))
# b =[(i, j, R[i,j]) for i, j in zip(*R.nonzero())]
# print("converted to desired list format at ", str(datetime.now()))


mf = MF(R, K=50, alpha=0.1, beta=0.2, iterations=1000)
mf.train()
# print(mf.full_matrix())

np.save("Data/full_matrix.npy", mf.full_matrix())