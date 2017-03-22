import pandas as pd

def findCorrelation(X, threshold = 0.9):
  '''
  Find pairwise correlations beyond threshhold
  This is not 'exact': it does not recalculate correlation after each step, and is therefore less expensive

  Inputs:
  X: pandas dataframe containing numeric values
  threshold: cutoff correlation threshold
  Returns:
  List of column names to filter where appropriate

  This is equivalent to findCorrelation() in the R caret package with exact = False
  '''

  corrMat = X.corr()
  colNames = X.columns.values.tolist()
  corrNames = list()
  row_count = 0

  for name in colNames:
    corrRows = (corrMat
                .iloc[row_count:]
                .loc[corrMat[name] >= threshold, name]
                .index
                .values
              )

    corrRows = [x for x in corrRows if x != name]
    avgCorr_curr = abs(corrMat[name]).mean()
    if len(corrRows) > 0:
      for i in corrRows:
        if abs(corrMat[i]).mean() > avgCorr_curr:
          corrNames.append(i)
        else:
          corrNames.append(name)
    row_count += 1

  return list(set(corrNames))



def nearZeroVariance(X, freqCut = 95 / 5, uniqueCut = 10):
  '''
  Determine predictors with near zero or zero variance.

  Inputs:
  X: pandas data frame
  freqCut: the cutoff for the ratio of the most common value to the second most common value
  uniqueCut: the cutoff for the percentage of distinct values out of the number of total samples
  Returns a tuple containing a list of column names: (zeroVar, nzVar)

  This provides the functionality of the R caret package nearZeroVariance() function.
 '''

  colNames = X.columns.values.tolist()
  freqRatio = dict()
  uniquePct = dict()
  nObs = X.shape[0]

  for name in colNames:
    counts = ( X[name]
              .value_counts()
              .sort_values(ascending = False)
              .values
             )

    if len(counts) == 1:
      freqRatio[name] = -1 # non defined
      uniquePct[name] = (len(counts) / nObs) * 100
      continue

    freqRatio[name] = counts[0] / counts[1]
    uniquePct[name] = (len(counts) / nObs) * 100

  zeroVar = list()
  nzVar = list()
  for k in uniquePct.keys():
    if freqRatio[k] == -1:
      zeroVar.append(k)
      nzVar.append(k)

    if uniquePct[k] < uniqueCut and freqRatio[k] > freqCut:
      nzVar.append(k)

  # consider updating to return a pd dataframe
  # | varname | uniquepct | freqratio | nvz | zv |

  return zeroVar, nzVar


