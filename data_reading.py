import pandas as pd
from random import randrange
from random import sample


def load_csv_dataset(filename):
    """Load the CSV file"""
    dataset = pd.read_csv(filename)
    # dataset = list(lines)
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]  # Convert String to Float numbers
    return dataset


def cross_validation_split(dataset, n_folds):
    """Split dataset into the k folds. Returns the list of k folds"""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def percentageSplit(dataset, trainingPer=80, testingPer=20):
    """Evaluate an algorithm using a cross validation split"""

    from numpy.core.tests.test_mem_overlap import xrange

    datasize = dataset.shape[0]
    train_size = int(datasize * trainingPer / 100)
    test_size = datasize - train_size

    indexes_to_remove = sample(xrange(datasize), test_size)
    indexes_to_keep = set(range(datasize)) - set(indexes_to_remove)

    train_set = dataset.take(list(indexes_to_keep))
    test_set = dataset.take(list(indexes_to_remove))

    return train_set, test_set


filename = 'Data/ml-20m/ratings.csv'
dataset = load_csv_dataset(filename)
print(len(dataset), dataset.shape)
x, y = percentageSplit(dataset)
