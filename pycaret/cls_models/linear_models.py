import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pycaret.utils.helpers import expand_grid


class GLM_L1():
  '''
  Wrapper class around sklearn.linear_model.LogisticRegression
  '''
  grid = pd.DataFrame({'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]})
  modelType = "Classification"
  model_descr = "Logistic regression with L1 penalty"

  def __init__(self, **kwargs):
    self.model = LogisticRegression(penalty = 'l1', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class GLM_L2():
  '''
  Wrapper class around sklearn.linear_model.LogisticRegression
  '''
  grid = pd.DataFrame({'C': [0.001, 0.003, 0.01, 0.3, 0.1, 0.3, 1, 3, 10, 30, 100]})
  modelType = "Classification"
  model_descr = "Logistic regression with L2 penalty"

  def __init__(self, **kwargs):
    self.model = LogisticRegression(penalty = 'l2', **kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class LDA():
  '''
  Wrapper class around sklearn.discriminant_analysis.LinearDiscriminantAnalysis
  '''
  grid = None
  modelType = "Classification"
  model_descr = "Linear discriminant analysis"

  def __init__(self, **kwargs):
    self.model = LinearDiscriminantAnalysis(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)
