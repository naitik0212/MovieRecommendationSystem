# Reference: https://github.com/alexvlis/movie-recommendation-system/blob/master/nn.py
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers

# Change the paths to data being used
moviepath = 'Data/ml-100k/movies.csv'
train_ratingpath = 'Data/ml-100k/train_ratings.csv'
test_ratingpath = 'Data/ml-100k/test_ratings.csv'


def create_mapping(path='Data/ml-100k/ratings.csv'):
    '''
    Creates a mapping of unique user ID with user ID from
    ratings.csv file. It creates a similar mapping for movieID.
    Returns both the mappings as dictionaries.
    '''
    ratings = pd.read_csv(path)
    unique_userIds = ratings.userId.unique()
    uniqueUserMappingDict = {j: i for i, j in enumerate(unique_userIds)}

    unique_movieIds = ratings.movieId.unique()
    uniqueMovieMappingDict = {j: i for i, j in enumerate(unique_movieIds)}

    n_users = int(ratings.userId.nunique())
    n_movies = int(ratings.movieId.nunique())

    return uniqueUserMappingDict, uniqueMovieMappingDict, n_users, n_movies


def inputgen(ratingPath, uniqueUserMappingDict, uniqueMovieMappingDict, n_users, n_movies):
    '''
    Generates the data matrix to be used by the
    neural network by using the mapping dictionaries
    '''
    ratings = pd.read_csv(ratingPath)
    ratings.userId = ratings.userId.apply(lambda x: uniqueUserMappingDict[x])
    ratings.movieId = ratings.movieId.apply(lambda x: uniqueMovieMappingDict[x])

    feature_matrix = np.zeros((n_users, n_movies))

    for i in range(ratings.shape[0]):
        feature_matrix[ratings.iloc[i, 0]][ratings.iloc[i, 1]] = ratings.iloc[i, 2]

    user_indices, movie_indices = (np.where(feature_matrix > 0))

    data_x = np.zeros((user_indices.shape[0], n_users + n_movies))
    data_y = np.zeros((user_indices.shape[0], 1))

    for i in range(user_indices.shape[0]):
        u = user_indices[i]
        m = movie_indices[i]
        data_x[i, u] = 1
        data_x[i, n_users + m] = 1

        score = feature_matrix[u, m]
        data_y[i] = score

    return data_x, data_y


def construct_model(n_users, n_movies):
    '''
    Returns a sequential model with 3 layers with each layer having
    exponentially less number of neurons than previous layer
    '''
    model = Sequential()
    input_shape = n_users + n_movies
    n_layers = 3

    model.add(Dense(4096, activation='relu', input_shape=(input_shape,)))

    n_neurons = int((np.exp(np.log(input_shape) / (n_layers + 2))))

    for i in range(n_layers - 1):
        input_shape = int(input_shape / n_neurons)
        model.add(Dense(input_shape, activation='relu'))
        model.add(Dropout(0.2))

    input_shape = int(input_shape / n_neurons)
    model.add(Dense(input_shape, activation='relu'))

    print(model.output_shape)

    # Output Layer
    model.add(Dense(1, activation='relu'))

    adam = optimizers.Adam(0.01, decay=.001)

    # Using Mean Square error as loss and adam optimizer
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    print(model.summary())
    return model


def training(model, train_x, train_y, model_number=0, learn_rate=0.01):
    '''
    Trains the model using train_x, train_y and saves checkpoints of the model at every epoch.
    '''

    # Making Checkpoints
    filepath = "nn_model_{}_lr_{}".format(model_number, learn_rate)
    filepath += "_{epoch:02d}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                                                 save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]
    print(train_x.shape)
    model.fit(train_x, train_y, batch_size=64, epochs=2, callbacks=callbacks_list, validation_split=0.2, verbose=1)


def test(test_x, test_y, model):
    '''
    Makes prediction on the test_x and model,
    Calculates error by comparing with test_y
    '''
    prediction = model.predict(test_x, verbose=True)

    error = prediction - test_y
    mean_square_error = np.mean(np.power(error, 2))

    print('The MSE of the model is {}'.format(mean_square_error))
    print('The RMSE of the model is {}'.format(np.sqrt(mean_square_error)))


uniqueUserMappingDict, uniqueMovieMappingDict, users, movies = create_mapping()
train_x, train_y = inputgen(ratingPath=train_ratingpath, uniqueUserMappingDict=uniqueUserMappingDict,
                            uniqueMovieMappingDict=uniqueMovieMappingDict, n_users=users, n_movies=movies)

model = construct_model(users, movies)
training(model, train_x, train_y)

test_x, test_y = inputgen(ratingPath=test_ratingpath, uniqueUserMappingDict=uniqueUserMappingDict,
                          uniqueMovieMappingDict=uniqueMovieMappingDict, n_users=users, n_movies=movies)
test(test_x, test_y, model)
