import itertools as it
import pandas as pd


def expand_grid(param_dict):
  '''
  perform a grid expansion (cartesian product)

  Inputs:
  param_dict: a dictionary with keys as parameter names, values
              as a list of paramter values
  Returns:
  pandas dataframe
  '''
  varNames = sorted(param_dict)
  tmp = [prod for prod in it.product(*(param_dict[varName] for varName in varNames))]
  df = pd.DataFrame(tmp)
  df.columns = varNames
  return df


