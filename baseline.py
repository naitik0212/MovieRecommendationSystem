import pandas as pd
import json
import math
import os.path

import data_reading
import db_helper

filename = 'Data/ml-20m/train_ratings.csv'
testfilename = 'Data/ml-20m/test_ratings.csv'
modelfilename = 'Data/ml-20m/baseline_model'


# filename = 'Data/ml-100k/train_ratings.csv'
# testfilename = 'Data/ml-100k/test_ratings.csv'
# modelfilename = 'Data/ml-100k/baseline_model'


def getTrainTestData():
    train, test = data_reading.getTrainTestData(filename, file_trainRating=filename, file_testRating=testfilename)
    return train, test


def getMean(dataset, columnName='rating'):
    return dataset[columnName].mean()


def getUniqueColumnsValues(dataset, columnName):
    return dataset[columnName].unique()


def loadModel(path=modelfilename):
    if not os.path.isfile(modelfilename):
        training()

    with open(modelfilename, 'r') as f:
        model = json.load(f)
    return model


def training():
    train, test = getTrainTestData()
    mu = getMean(train, columnName='rating')
    userRatingDeviation = train[['userId', 'rating']].groupby('userId').mean()
    averageMovieRatings = train[['movieId', 'rating']].groupby('movieId').mean()

    model = {
        'mean': mu,
        'userDeviation': userRatingDeviation.to_dict(),
        'movieRating': averageMovieRatings.to_dict()
    }

    with open(modelfilename, 'w+') as f:
        json.dump(model, f)


def testing():
    print('Testing...')
    train, test = getTrainTestData()
    print('Loading baseline model')
    model = loadModel(modelfilename)
    print('Baseline model loaded')

    mu = model['mean']
    userDeviation = model['userDeviation']
    movieRating = model['movieRating']

    def baselineEstimate(row):
        if str(int(row['userId'])) in userDeviation['rating'] and str(int(row['movieId'])) in movieRating['rating']:
            rating = mu + (userDeviation['rating'][str(int(row['userId']))] - mu) + (
                    movieRating['rating'][str(int(row['movieId']))] - mu)
            return db_helper.roundRatings(rating)
        elif str(int(row['userId'])) in userDeviation['rating'] and str(int(row['movieId'])) not in movieRating[
            'rating']:
            rating = mu + (userDeviation['rating'][str(int(row['userId']))] - mu) + 0
            return db_helper.roundRatings(rating)
        elif str(int(row['userId'])) not in userDeviation['rating'] and str(int(row['movieId'])) in movieRating[
            'rating']:
            rating = mu + 0 + (movieRating['rating'][str(int(row['movieId']))] - mu)
            return db_helper.roundRatings(rating)
        return mu

    print('Calculating rating estimates...')
    test['ratingEstimate'] = test.apply(baselineEstimate, axis=1)
    print('Rating estimation done')
    print('Calculating RMSE')
    test['diff'] = test.apply(lambda x: (x['rating'] - x['ratingEstimate']) ** 2, axis=1)

    squareSum = test['diff'].sum()
    total_testcase = test.shape[0]
    RMSE = math.sqrt(squareSum / total_testcase)
    print('Root mean square error: %s' % RMSE)
    print('Testing completed...')


def baselineEstimate(model, user, movie):
    mu = model['mean']
    userDeviation = model['userDeviation']
    movieRating = model['movieRating']

    if str(int(user)) in userDeviation['rating'] and str(int(movie)) in movieRating['rating']:
        rating = mu + (userDeviation['rating'][str(int(user))] - mu) + (movieRating['rating'][str(int(movie))] - mu)
        return db_helper.roundRatings(rating)
    elif str(int(user)) in userDeviation['rating'] and str(int(movie)) not in movieRating['rating']:
        rating = mu + (userDeviation['rating'][str(int(user))] - mu) + 0
        return db_helper.roundRatings(rating)
    elif str(int(user)) not in userDeviation['rating'] and str(int(movie)) in movieRating['rating']:
        rating = mu + 0 + (movieRating['rating'][str(int(movie))] - mu)
        return db_helper.roundRatings(rating)
    return mu


if __name__ == '__main__':
    testing()
    # baselinemodel = loadModel()
    # print(baselineEstimate(baselinemodel, 1, 44245))
