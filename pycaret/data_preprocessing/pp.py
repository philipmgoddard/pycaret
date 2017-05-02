from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

'''
Aim: to wrap any preprocessing functionality with a standard interface.
Any preprocessing method could be added. Requirement is that it has a fit method
that takes a pandas dataframe of features, and a transform method that applies the
fitted transformation, returning a dataframe of transformed features.
'''

class CS_Process():
  '''
  Wrapper class to provide mean centering and scaling.
  This implementation is a wrapper around sklearn.preprocessing.StandardScaler
  '''

  def __init__(self, **kwargs):
    self.pp_method = StandardScaler(**kwargs)

  def fit(self, x):
    self.pp_method.fit(x)

  def transform(self, x):
    colnames = [n for n in x.columns]
    trans = self.pp_method.transform(x)
    return pd.DataFrame(trans, columns=colnames)


class PCA_Process():
  '''
  Wrapper class to provide mean centering and scaling.
  This implementation is a wrapper around sklearn.decomposition.PCA

  Highly recommended that data is centered and scaled before applying PCA transformation.
  '''

  def __init__(self, **kwargs):
    self.pp_method = PCA(**kwargs)

  def fit(self, x):
    self.pp_method.fit(x)

  def transform(self, x):
    if self.pp_method.n_components is None:
      colnames = ['comp_'+ str(n+1) for n in range(x.shape[1])]
    else:
      colnames = ['comp_'+ str(n+1) for n in range(self.pp_method.n_components)]

    trans = self.pp_method.transform(x)
    return pd.DataFrame(trans, columns=colnames)




