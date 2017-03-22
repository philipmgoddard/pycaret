import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from pycaret.utils.helpers import expand_grid


class LM():
  grid = None
  modelType = "Regression"
  model_descr = "Linear model for regression"

  def __init__(self, **kwargs):
    self.model = LinearRegression(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class Ridge_Reg():
  grid = pd.DataFrame({'alpha' : [0.001, 0.003, 0.01, 0.3, 0.1, 0.3, 1]})
  modelType = 'Regression'
  model_descr = 'Ridge regression'

  def __init__(self, **kwargs):
    self.model = Ridge(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)


class ENet():
  grid = expand_grid({'alpha': [0.001, 0.003, 0.01, 0.3, 0.1, 0.3, 1], \
                          'l1_ratio' : [0.001, 0.25, 0.5, 0.75, 0.99]})
  modelType = "Regression"
  model_descr = "Elastic net"

  def __init__(self, **kwargs):
    self.model = ElasticNet(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x, **kwargs)
