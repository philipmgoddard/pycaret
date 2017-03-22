import numpy as np
import pandas as pd


def downSample(X, y, seed = 1234):
  '''
  Downsample to balance classes
  Inputs:
  X: a pandas data frame
  y: string the column of X that is the outcome
  seed random seed
  Returns:
  Pandas data frame where tuples from the majority class are randomly
  removed to balance the sample with respect to the outcome
  '''
  np.random.seed(seed)

  # determine the outcome categories
  levels = X[y].unique()
  # how many rows in each category
  count = dict()
  for level in levels:
    count[level] = count.get(level, 0) + len(X[X[y] == level])

  # determine what is the outcome with the min number of rows
  minOutcome = min(count.keys(), key = (lambda k: count[k]))
  nSample = count[minOutcome]

  # now need to sample rows index from all outcomes
  toKeep = (X[X[y] == minOutcome].index.values)
  for level in levels[levels != minOutcome]:
    indexVals = X[X[y] == level].index.values
    toKeep = np.append(toKeep,
                       np.random.choice(indexVals, nSample, replace = False))

  # subset and return the data frame
  return X.ix[toKeep].reset_index()


def upSample(X, y, seed = 1234):
  '''
  Upsample to balance classes
  Inputs:
  X: a pandas data frame
  y: the column of X that is the outcome
  seed: random seed
  Returns:
  Pandas data frame where tuples from the minority class are randomly
  replicated to balance the sample with respect to the outcome
  '''

  # set seed
  np.random.seed(seed)

  # determine the outcome categories
  levels = X[y].unique()

  # how many rows in each category
  count = dict()
  for level in levels:
    count[level] = count.get(level, 0) + len(X[X[y] == level])

  # determine what is the outcome with the max number of rows
  maxOutcome = max(count.keys(), key = (lambda k: count[k]))
  nSample = count[maxOutcome]

  # now need to sample rows index from all outcomes
  toKeep = (X[X[y] == maxOutcome].index.values)
  for level in levels[levels != maxOutcome]:
    indexVals = X[X[y] == level].index.values
    upSamps = np.random.choice(indexVals,
                               (nSample - len(indexVals)),
                               replace = True)
    indexVals = np.append(upSamps, indexVals)
    toKeep = np.append(toKeep, indexVals)

  # subset and return the data frame
  return X.ix[toKeep].reset_index()
