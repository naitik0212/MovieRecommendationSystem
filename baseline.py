import pandas as pd
import json
import math
import os.path

import data_reading
import db_helper

filename = 'Data/ml-20m/ratings.csv'
modelfilename = 'Data/ml-20m/baseline_model'
#train, test = data_reading.getTrainTestData(filename)


def getMean(dataset, columnName='rating'):
    return dataset[columnName].mean()


def getUniqueColumnsValues(dataset, columnName):
    return dataset[columnName].unique()


def loadModel(path=modelfilename):
    print("hello")
    if not os.path.isfile(modelfilename):
        training()

    with open(modelfilename, 'r') as f:
        model = json.load(f)
    return model


def training():
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
    model = loadModel(modelfilename)

    mu = model['mean']
    userDeviation = model['userDeviation']
    movieRating = model['movieRating']

    def baselineEstimate(row):
        rating = mu + (userDeviation['rating'][str(int(row['userId']))] - mu) + (
                movieRating['rating'][str(int(row['movieId']))] - mu)
        return db_helper.roundRatings(rating)

    test['ratingEstimate'] = test.apply(baselineEstimate, axis=1)
    test['diff'] = test.apply(lambda x: (x['rating'] - x['ratingEstimate']) ** 2, axis=1)

    squareSum = test['diff'].sum()
    total_testcase = test.shape[0]
    RMSE = math.sqrt(squareSum / total_testcase)
    print(RMSE)


def baselineEstimate(model, user, movie):
    mu = model['mean']
    userDeviation = model['userDeviation']
    movieRating = model['movieRating']

    rating = mu + (userDeviation['rating'][str(int(user))] - mu) + (movieRating['rating'][str(int(movie))] - mu)
    return db_helper.roundRatings(rating)


testing()
# baselinemodel = loadModel()
# print(baselineEstimate(baselinemodel, 71365,5528))
