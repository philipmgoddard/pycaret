import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from pycaret.utils.helpers import expand_grid


class KNN_Reg():
  '''
  sklearn.neighbors.KNeighborsRegressor
  '''
  grid = pd.DataFrame({'n_neighbors' :[1, 3, 5, 7, 9]})
  modelType = "Regression"
  model_descr = "K nearest neighbors model for regression"

  def __init__(self, **kwargs):
    self.model = KNeighborsRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class SVR_L():
  '''
  Wrapper around sklearn.svm.SVR
  '''
  grid = pd.DataFrame({'C': [0.001, 0.003, 0.01, 0.3, 1.0]})
  modelType = "Regression"
  model_descr = "SVM with RBF kernal for regression"

  def __init__(self, **kwargs):
    self.model = SVR(kernel = 'linear', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class SVR_RBF():
  '''
  Wrapper around sklearn.svm.SVR
  '''
  grid = expand_grid({'C': [0.001, 0.003, 0.01, 0.3, 1.0],
                     'gamma' : [0.01, 0.03, 0.1, 0.3]})
  modelType = "Regression"
  model_descr = "SVM with RBF kernal for regression"

  def __init__(self, **kwargs):
    self.model = SVR(kernel = 'rbf', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class SVR_P():
  '''
  Wrapper around sklearn.svm.SVR
  '''
  grid = expand_grid({'C': [0.01, 0.03, 0.1, 0.3,  1.0],
                     'gamma' : [0.01, 0.03, 0.1, 0.3],
                     'degree' : [2, 3, 4]})
  modelType = "Regression"
  model_descr = "SVM with polynomial kernal for regression"

  def __init__(self, **kwargs):
    self.model = SVR(kernel = 'poly', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class NNet_Reg():
  '''
  Wrapper around sklearn.neural_network.MLPRegressor
  '''
  grid = expand_grid({'hidden_layer_sizes': [50, 100, 200],
                      'alpha' : [0.0001, 0.001, 0.01, 0.01, 1.0]})
  modelType = "Regression"
  model_descr = "Single layered neural network regressor"

  def __init__(self, **kwargs):
    self.model = MLPRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)
