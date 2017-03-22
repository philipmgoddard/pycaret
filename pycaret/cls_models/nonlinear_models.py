import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from pycaret.utils.helpers import expand_grid

class KNN_Cls():
  '''
  Wrapper class around sklearn.neighbors.KNeighborsClassifier
  '''
  grid = pd.DataFrame({'n_neighbors' :[1, 3, 5, 7, 9]})
  modelType = "Classification"
  model_descr = "K nearest neighbors model for classification"

  def __init__(self, **kwargs):
    self.model = KNeighborsClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class NB_Gauss():
  '''
  Wrapper class around sklearn.naive_bayes.GaussianNB
  '''
  grid = None
  modelType = "Classification"
  model_descr = "Gaussian naive bayes"

  def __init__(self, **kwargs):
    self.model = GaussianNB(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class SVC_L():
  '''
  Wrapper class around sklearn.svm.SVC
  '''
  grid = pd.DataFrame({'C': [0.001, 0.003, 0.01, 0.3, 1.0]})
  modelType = "Classification"
  model_descr = "SVM with Linear kernal for classification"

  def __init__(self, **kwargs):
    self.model = SVC(kernel = 'linear', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      if self.model.probability is False:
        print('model probabilty is False')
        return
      else:
        return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class SVC_RBF():
  '''
  Wrapper class around sklearn.svm.SVC
  '''
  grid = expand_grid({'C': [ 0.3, 1.0, 3.0, 10.0],
                     'gamma' : [ 0.1, 0.3, 1.0, 3.0]})
  modelType = "Classification"
  model_descr = "SVM with RBF kernal for classification"

  def __init__(self, **kwargs):
    self.model = SVC(kernel = 'rbf', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      if self.model.probability is False:
        print('model probabilty is False')
        return
      else:
        return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class SVC_P():
  '''
  Wrapper class around sklearn.svm.SVC
  '''
  grid = expand_grid({'C': [0.01, 0.03, 0.1, 0.3,  1.0],
                     'gamma' : [0.01, 0.03, 0.1, 0.3],
                     'degree' : [2, 3, 4]})
  modelType = "Classification"
  model_descr = "SVM with polynomial kernal for classification"

  def __init__(self, **kwargs):
    self.model = SVC(kernel = 'poly', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      if self.model.probability is False:
        print('model probabilty is False')
        return
      else:
        return self.model.predict_proba(x)
    else:
      return self.model.predict(x)

class NNet_Cls():
  '''
  Wrapper class around sklearn.neural_net.MLPClassifier
  '''
  grid = expand_grid({'hidden_layer_sizes': [50, 100, 200],
                      'alpha' : [0.0001, 0.001, 0.01, 0.01, 1.0]})
  modelType = "Classification"
  model_descr = "Single layered neural network classifier"

  def __init__(self, **kwargs):
    self.model = MLPClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


