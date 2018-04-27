import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers

moviepath = 'Data/ml-100k/movies.csv'
train_ratingpath = 'Data/ml-100k/train_ratings.csv'
test_ratingpath = 'Data/ml-100k/test_ratings.csv'


def create_mapping(path='Data/ml-100k/ratings.csv'):
    ratings = pd.read_csv(path)
    unique_userIds = ratings.userId.unique()
    uniqueUserMappingDict = {j: i for i, j in enumerate(unique_userIds)}

    unique_movieIds = ratings.movieId.unique()
    uniqueMovieMappingDict = {j: i for i, j in enumerate(unique_movieIds)}

    n_users = int(ratings.userId.nunique())
    n_movies = int(ratings.movieId.nunique())

    return uniqueUserMappingDict, uniqueMovieMappingDict, n_users, n_movies


def inputgen(ratingPath, uniqueUserMappingDict, uniqueMovieMappingDict, n_users, n_movies):
    ratings = pd.read_csv(ratingPath)

    ratings.userId = ratings.userId.apply(lambda x: uniqueUserMappingDict[x])

    ratings.movieId = ratings.movieId.apply(lambda x: uniqueMovieMappingDict[x])

    feature_matrix = np.zeros((n_users, n_movies))
    print(ratings.shape)
    for i in range(ratings.shape[0]):
        feature_matrix[ratings.iloc[i, 0]][ratings.iloc[i, 1]] = ratings.iloc[i, 2]

    # n_rows = feature_matrix.shape[0]
    # n_cols = feature_matrix.shape[1]

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


def split_data(train_data, label_data):
    split_percent = (int)(0.80 * len(train_data))
    train_x = train_data[:split_percent, :]
    train_y = label_data[:split_percent, :]
    test_x = train_data[split_percent:, :]
    test_y = label_data[split_percent:, :]
    return train_x, test_x, train_y, test_y


def construct_model(train_data, label_data, m, n):
    model = Sequential()
    input_size = m + n
    #     input_size = 4096
    num_layers = 3
    # add the first layer
    model.add(Dense(4096, activation='relu', input_shape=(input_size,)))

    exponential_decrease = int((np.exp(np.log(input_size) / (num_layers + 2))))
    print(exponential_decrease)
    for i in range(num_layers - 1):
        input_size = int(input_size / exponential_decrease);
        model.add(Dense(input_size, activation='relu'))
        model.add(Dropout(0.4))

    input_size = int(input_size / exponential_decrease);
    model.add(Dense(input_size, activation='relu'))

    print(model.output_shape)
    # one hot encoded output
    model.add(Dense(1, activation='relu'))

    # model says they optimized the log loss error

    adam = optimizers.Adam(0.01, decay=.001)
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])

    return model


def train_model(model, train_x, train_y, model_number=0, learn_rate=0.01):
    '''
    Trains the model. Saves checkpoints of the model at every epoch.
    I personally just stop training when I find that the loss function has barely changed. Since it takes
    so long to perform each epoch on my computer, I just keep running a 20 epoch train, stop it when I
    have to, then train again later.
    Param:
        model_number - Just changes the filename that the model is saved to. 
                       Don't want to overwrite good save files during training, do you?
    Note: these checkpoints are 1GB each.
    '''
    # lets make checkpoints
    filepath = "nn_model_{}_lr_{}".format(model_number, learn_rate)
    filepath += "_{epoch:02d}.hdf5"

    print('learn_rate = {}'.format(learn_rate))
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                                                 save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint]

    model.fit(train_x, train_y, batch_size=64, epochs=5, callbacks=callbacks_list, validation_split=0.2, verbose=1)


def test(test_x, test_y, model):
    pred_scores = model.predict(test_x, verbose=True)

    # pred_scores = pred_scores.argmax(axis=1)
    # true_scores    = true_scores.argmax(axis=1)

    # get Accuracy
    num_correct = np.sum(pred_scores == test_y)
    accuracy = num_correct / pred_scores.shape[0] * 100

    # get MSE
    error = pred_scores - test_y
    mse = np.mean(np.power(error, 2))

    print('The accuracy of the model is {}%'.format(accuracy))
    print('The mean squared error of the model is {}'.format(mse))


uniqueUserMappingDict, uniqueMovieMappingDict, users, movies = create_mapping()
train_x, train_y = inputgen(ratingPath=train_ratingpath, uniqueUserMappingDict=uniqueUserMappingDict,
                            uniqueMovieMappingDict=uniqueMovieMappingDict, n_users=users, n_movies=movies)
# train_x, test_x, train_y, test_y = split_data(train_x, train_y)
model = construct_model(train_x, train_y, users, movies)
train_model(model, train_x, train_y)

test_x, test_y = inputgen(ratingPath=test_ratingpath, uniqueUserMappingDict=uniqueUserMappingDict,
                          uniqueMovieMappingDict=uniqueMovieMappingDict, n_users=users, n_movies=movies)
test(test_x, test_y, model)
