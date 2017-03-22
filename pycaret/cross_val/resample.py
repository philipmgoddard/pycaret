import numpy as np

def kfold(index, k = 10, seed = 1234):
  '''
  k fold cross validation
  index: index values (np array) from pandas data frame containing training sample
  k: number of folds in cv
  seed: for setting random seed.

  yields: indicies for current fold.
  '''
  np.random.seed(seed = seed)
  len_index = len(index)
  ind = np.arange(len_index)
  np.random.shuffle(ind)

  for fold in np.array_split(ind, k):
    yield fold


def repeated_cv(index, repeats = 3, k = 10, seed = 1234):
  '''
  repeated k fold cross validation
  repeats: number of repeats
  k: number of folds in cv
  index: index from pandas data frame containing training sample
  k: number of folds in cv
  seed: for setting random seed.

  yields: indicies for current fold.
  '''
  np.random.seed(seed = seed)
  len_index = len(index)
  ind = np.arange(len_index)
  np.random.shuffle(ind)

  for i in np.arange(repeats):
    for fold in np.array_split(ind, k):
      yield fold
    np.random.shuffle(ind)


def boot(index, number = 25, seed = 1234):
  '''
  bootstrap resampling
  index: index from pandas data frame containing training sample
  number: number of bootstrap resamples
  seed: for setting random seed.

  yields: indicies for current sample.
  '''
  np.random.seed(seed=seed)
  len_index = len(index)
  ind = np.arange(len_index)
  np.random.shuffle(ind)

  for i in np.arange(number):
    yield np.random.choice(ind, size = len_index, replace = True, p = None)


