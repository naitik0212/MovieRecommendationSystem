import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from baseline import loadModel
from baseline import baselineEstimate

import os

RANGE_MIN = 1
RANGE_MAX = 2

DATASET_ROOT_PATH = os.path.join(os.getcwd(), './', '')
OUT_PUT = []


def readcsv(name):
    return pd.read_csv(os.path.join(DATASET_ROOT_PATH, name))


def testing(k):
    print("testing")
    similarfilename = "similarityMatrix" + str(k) + ".npy"
    similarityMatrix = np.load("Data/ml-20m/" + similarfilename)
    indexMatrix = np.load("Data/ml-20m/movieindex.npy")
    baselinemodel = loadModel()
    ratings = readcsv('Data/ml-20m/train_ratings.csv')
    testData = readcsv('Data/ml-20m/test_ratings.csv')
    testData = testData.sort_values('userId', ascending=False)
    uniqueUserId = testData.userId.unique()
    sumrmse = 0
    i = 0
    count = 0
    for id in np.nditer(uniqueUserId):
        # print(id)
        trainingDataExists = True

        ratingU = ratings.loc[ratings['userId'] == id]
        ratingU = ratingU.drop('timestamp', 1)

        if len(ratingU) == 0:
            trainingDataExists = False

        testUserData = testData.loc[testData['userId'] == id]
        for index, row in testUserData.iterrows():
            count += 1
            uid = row['userId']
            mid = row['movieId']
            if trainingDataExists == True:
                rxi = calculateRatingIX(uid, mid, similarityMatrix, indexMatrix, ratingU, baselinemodel)
            else:
                rxi = baselineEstimate(baselinemodel, uid, mid)

            sumrmse += np.square(row['rating'] - rxi)

    rmse = np.sqrt(sumrmse / count)
    print(rmse)
    return rmse


def calculateRatingIX(uid, mid, similarityMatrix, indexMatrix, ratingU, baselinemodel):
    index = np.where(indexMatrix == mid)[0][0]
    similarity = similarityMatrix[index]
    similarityDF = pd.DataFrame(similarity.reshape(-1, len(similarity)))
    similarityDF = similarityDF.transpose()
    indexDF = pd.DataFrame(indexMatrix.reshape(-1, len(indexMatrix)))
    indexDF = indexDF.transpose()
    newDF = pd.concat([indexDF, similarityDF], axis=1)
    newDF.columns = ['movieId', 'similarity']

    newRating = ratingU.merge(newDF, left_on='movieId', right_on='movieId')
    newRating = newRating.sort_values('similarity', ascending=False)
    NUM_SIMILAR = min(10, len(ratingU))
    top = newRating.head(n=NUM_SIMILAR)

    similaritySum = top['similarity'].sum()

    bxi = baselineEstimate(baselinemodel, uid, mid)
    sum = 0
    for index, row in top.iterrows():
        bxj = baselineEstimate(baselinemodel, row['userId'], row['movieId'])
        rxj = row['rating']
        sij = row['similarity']
        sum += (rxj - bxj) * sij

    rxi = bxi + sum / similaritySum

    return rxi


def generate_model():
    print("Generating tags model")

    merged = readcsv('Data/ml-20m/new_tags_generes.csv')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    group_data = pd.DataFrame(merged.groupby(['movieId', 'genres'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['movieId'], 'genres')
    term_vector.index.names = ['label']
    count_df = generate_idf(term_vector)
    term_vector = term_vector.fillna(0)
    term_vector = generate_TF(term_vector)

    tf_idf = term_vector.copy(deep=True)
    tf_idf = tf_idf.mul(count_df.ix[2], axis='columns')
    indexValues = tf_idf.index.values.tolist()
    indexValues = np.array(indexValues)
    np.save("Data/ml-20m/movieindex.npy", indexValues)

    temp = tf_idf.as_matrix()
    svd = TruncatedSVD(n_components=100)
    x = svd.fit_transform(temp)
    similarityMatrix = cosine_similarity(x)
    np.save("Data/ml-20m/similarityMatrix.npy", similarityMatrix)


## given a term vector it generates Term frequency for all the documents
## in the dataframe

def generate_TF(term_vector):
    term_vector['total_freq'] = term_vector.sum(axis=1)
    columns = term_vector.columns.tolist()
    columns.remove('total_freq')
    term_vector = term_vector[columns].div(term_vector.total_freq, axis=0)
    return term_vector


## given a term vector it generates Inverse Document frequency for all the documents
## in the dataframe
def generate_idf(term_vector):
    count = term_vector.count(axis='index')
    count_df = count.to_frame()
    count_df['total_docs'] = len(term_vector.index)
    count_df.columns = ['tag_count', 'total_docs']
    count_df['total_docs'] = pd.to_numeric(count_df['total_docs'])
    count_df['tag_count'] = pd.to_numeric(count_df['tag_count'])
    count_df['idf'] = np.log(count_df.total_docs / count_df.tag_count)
    count_df = count_df.T
    return count_df


#
# if not os.path.exists("Data/ml-20m/similarityMatrix.npy"):
#     generate_model()


K = [30, 50, 100, 200]
RMSE = [testing(k) for k in K]

print(RMSE)
