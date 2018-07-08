import pandas as pd
from random import randrange
import numpy as np

np.random.seed(20)

file_trainRating = 'Data/ml-20m/train_ratings.csv'
file_testRating = 'Data/ml-20m/test_ratings.csv'


def load_csv_dataset(filename):
    """Load the CSV file"""
    dataset = pd.read_csv(filename)
    # dataset = list(lines)
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]  # Convert String to Float numbers
    return dataset


def getTrainTestData(filename, trainingPer=80, testingPer=20, file_trainRating=file_trainRating,
                     file_testRating=file_testRating):
    """
    Split the data set into training and testing percentage.
    """
    import os.path
    from numpy.core.tests.test_mem_overlap import xrange

    if os.path.isfile(file_trainRating) and os.path.isfile(file_testRating):
        train_set = pd.read_csv(file_trainRating)
        test_set = pd.read_csv(file_testRating)
        return train_set, test_set
    else:
        dataset = load_csv_dataset(filename)
        datasize = dataset.shape[0]
        train_size = int(datasize * trainingPer / 100)
        test_size = datasize - train_size

        indexes_to_remove = np.random.choice(xrange(datasize), test_size, replace=False).tolist()
        indexes_to_keep = set(xrange(datasize)) - set(indexes_to_remove)

        train_set = dataset.take(list(indexes_to_keep))
        test_set = dataset.take(list(indexes_to_remove))
        saveDataToFile(file_trainRating, train_set)
        saveDataToFile(file_testRating, test_set)

        return train_set, test_set


def saveDataToFile(filename, data):
    data.to_csv(filename, index=False)


filename = 'Data/ml-20m/ratings.csv'

train, test = getTrainTestData(filename)
