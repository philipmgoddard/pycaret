import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor

import pycaret.reg_models.linear_models as lin_mod
import pycaret.reg_models.nonlinear_models as nonlin_mod
import pycaret.reg_models.trees_rules as tree_rule

class RegModelsTestCase(unittest.TestCase):
  '''
  Basic test for regresion models: ensure object instantiated correctly
  '''

  def test_LM(self):
    self.assertTrue(isinstance(lin_mod.LM().model, LinearRegression))

  def test_Ridge(self):
    self.assertTrue(isinstance(lin_mod.Ridge_Reg().model, Ridge))

  def test_ENet(self):
    self.assertTrue(isinstance(lin_mod.ENet().model, ElasticNet))

  def test_KNN_Reg(self):
    self.assertTrue(isinstance(nonlin_mod.KNN_Reg().model, KNeighborsRegressor))

  def test_RF_Reg(self):
    self.assertTrue(isinstance(tree_rule.RF_Reg().model, RandomForestRegressor))

  def test_Ada_Reg(self):
    self.assertTrue(isinstance(tree_rule.Ada_Reg().model, AdaBoostRegressor))

  def test_SVR_L(self):
    self.assertTrue(isinstance(nonlin_mod.SVR_L().model, SVR))

  def test_SVR_RBF(self):
    self.assertTrue(isinstance(nonlin_mod.SVR_RBF().model, SVR))

  def test_SVR_P(self):
    self.assertTrue(isinstance(nonlin_mod.SVR_P().model, SVR))

  def test_NNet_Reg(self):
    self.assertTrue(isinstance(nonlin_mod.NNet_Reg().model, MLPRegressor))

  def test_CART_Reg(self):
    self.assertTrue(isinstance(tree_rule.CART_Reg().model, DecisionTreeRegressor))

  def test_GBM_Reg(self):
    self.assertTrue(isinstance(tree_rule.GBM_Reg().model, GradientBoostingRegressor))

  def test_GBM_Reg(self):
    self.assertTrue(isinstance(tree_rule.XGB_Reg().model, XGBRegressor))


if __name__ == "__main__":
  unittest.main()
