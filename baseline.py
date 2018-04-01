import pandas as pd
import json
import math

import data_reading
import db_helper

filename = 'Data/ml-20m/ratings.csv'
modelfilename = 'Data/ml-20m/baseline_model'
train, test = data_reading.getTrainTestData(filename)


def getMean(dataset, columnName='rating'):
    return dataset[columnName].mean()


def getUniqueColumnsValues(dataset, columnName):
    return dataset[columnName].unique()


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

    with open(modelfilename, 'r') as f:
        model = json.load(f)

    mu = model['mean']
    userDeviation = model['userDeviation']
    movieRating = model['movieRating']

    def baselineEstimate(row):
        # print(row)
        rating = mu + (userDeviation['rating'][str(int(row['userId']))] - mu) + (movieRating['rating'][str(int(row['movieId']))] - mu)
        return db_helper.roundRatings(rating)

    test['ratingEstimate'] = test.apply(baselineEstimate, axis=1)

    test['diff'] = test.apply(
        lambda x: (x['rating'] - x['ratingEstimate'])**2, axis=1)
    # print(test)
    square = test['diff'].sum()
    total_testcase = test.shape[0]
    RMS = math.sqrt(square/total_testcase)
    print(RMS)



# training()
testing()
