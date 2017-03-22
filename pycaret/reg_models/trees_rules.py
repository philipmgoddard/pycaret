import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

from pycaret.utils.helpers import expand_grid

class RF_Reg():
  '''
  Wrapper around sklearn.ensemble.RandomForestRegressor
  '''
  grid = pd.DataFrame({'max_features': [3, 5, 7, 9]})
  modelType = "Regression"
  model_descr = "Random forest regressor"

  def __init__(self, **kwargs):
    self.model = RandomForestRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x)


class Ada_Reg():
  '''
  Wrapper around sklearn.ensemble.AdaBoostRegressor
  '''
  grid = expand_grid({'n_estimators': [50, 100, 200],
                       'learning_rate': [0.1, 0.3, 1]})
  modelType = "Regression"
  model_descr = "Adaboost tree regressor"

  def __init__(self, **kwargs):
    self.model = AdaBoostRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x)


class CART_Reg():
  '''
  Wrapper around sklearn.tree.DecisionTreeRegressor
  '''
  grid = pd.DataFrame({'max_depth': [3, 5, 7, 9]})
  modelType = "Regression"
  model_descr = "Decision tree for regression"

  def __init__(self, **kwargs):
    self.model = DecisionTreeRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x)


class GBM_Reg():
  '''
  Wrapper around sklearn.ensemble.GradientBoostingRegressor
  '''
  grid = expand_grid({'n_estimators': [50, 100, 200],
                       'learning_rate': [0.1, 0.3, 1]})
  modelType = "Regression"
  model_descr = "Gradient Boosting Machine (GBM) Regressor"

  def __init__(self, **kwargs):
    self.model = GradientBoostingRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x)


class XGB_Reg():
  '''
  Wrapper around xgboost.sklearn.XGBRegressor
  '''
  grid = expand_grid({'max_depth': [3, 5, 7],
                       'learning_rate' : [0.03, 0.1, 0.3],
                       'n_estimators' : [100, 200, 300]})
  modelType = "Regression"
  model_descr = "XGB trees for regression"

  def __init__(self, **kwargs):
    self.model = XGBRegressor(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    return self.model.predict(x)
