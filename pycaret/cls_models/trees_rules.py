import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from pycaret.utils.helpers import expand_grid


class RF_Cls():
  '''
  Wrapper class around sklearn.ensemble.RandomForestClassifier
  '''
  grid = pd.DataFrame({'max_features': [3, 5, 7, 9]})
  modelType = "Classification"
  model_descr = "Random forest classifier"

  def __init__(self, **kwargs):
    self.model = RandomForestClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class Ada_Cls():
  '''
  Wrapper class around sklearn.ensemble.AdaBoostClassifier
  '''
  grid = expand_grid({'n_estimators': [50, 100, 200],
                       'learning_rate': [0.1, 0.3, 1]})
  modelType = "Classification"
  model_descr = "Adaboost classifier"

  def __init__(self, **kwargs):
    self.model = AdaBoostClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class GBM_Cls():
  '''
  Wrapper class around sklearn.ensemble.GradientBoostingClassifier
  '''
  grid = expand_grid({'n_estimators': [50, 100, 200],
                       'learning_rate': [0.1, 0.3, 1]})
  modelType = "Classification"
  model_descr = "Gradient Boosting Machine (GBM) Classifier"

  def __init__(self, **kwargs):
    self.model = GradientBoostingClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class CART_Cls():
  '''
  Wrapper class around sklearn.tree.DecisionTreeClassifier
  '''
  grid = pd.DataFrame({'max_depth': [3, 5, 7, 9]})
  modelType = "Classification"
  model_descr = "Decision tree for classification"

  def __init__(self, **kwargs):
    self.model = DecisionTreeClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)


class XGB_Cls():
  '''
  Wrapper class around xgboost.sklearn.XGBClassifier
  '''
  grid = expand_grid({'max_depth': [3, 5, 7],
                       'learning_rate' : [0.03, 0.1, 0.3],
                       'n_estimators' : [100, 200, 300]})
  modelType = "Classification"
  model_descr = "XGB trees for classification"

  def __init__(self, **kwargs):
    self.model = XGBClassifier(**kwargs)

  def train(self, x, y):
    self.model.fit(x, y)

  def predict(self, x, **kwargs):
    if kwargs.get('predict_proba'):
      return self.model.predict_proba(x)
    else:
      return self.model.predict(x)
