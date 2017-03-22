import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import pycaret.cls_models.linear_models as lin_mod
import pycaret.cls_models.nonlinear_models as nonlin_mod
import pycaret.cls_models.trees_rules as tree_rule


class ClsModelsTestCase(unittest.TestCase):
  '''
  Basic tests to ensure classification models are correctly instantiated
  '''

  def test_GLM_L1(self):
    self.assertTrue(isinstance(lin_mod.GLM_L1().model, LogisticRegression))

  def test_GLM_L2(self):
    self.assertTrue(isinstance(lin_mod.GLM_L2().model, LogisticRegression))

  def test_KNN_Cls(self):
    self.assertTrue(isinstance(nonlin_mod.KNN_Cls().model, KNeighborsClassifier))

  def test_NB_Gauss(self):
    self.assertTrue(isinstance(nonlin_mod.NB_Gauss().model, GaussianNB))

  def test_RF_Cls(self):
    self.assertTrue(isinstance(tree_rule.RF_Cls().model, RandomForestClassifier))

  def test_Ada_Cls(self):
    self.assertTrue(isinstance(tree_rule.Ada_Cls().model, AdaBoostClassifier))

  def test_SVC_L(self):
    self.assertTrue(isinstance(nonlin_mod.SVC_L().model, SVC))

  def test_SVC_RBF(self):
    self.assertTrue(isinstance(nonlin_mod.SVC_RBF().model, SVC))

  def test_SVC_P(self):
    self.assertTrue(isinstance(nonlin_mod.SVC_P().model, SVC))

  def test_NNet_Cls(self):
    self.assertTrue(isinstance(nonlin_mod.NNet_Cls().model, MLPClassifier))

  def test_GBM_Cls(self):
    self.assertTrue(isinstance(tree_rule.GBM_Cls().model, GradientBoostingClassifier))

  def test_CART_Cls(self):
    self.assertTrue(isinstance(tree_rule.CART_Cls().model, DecisionTreeClassifier))

  def test_LDA(self):
    self.assertTrue(isinstance(lin_mod.LDA().model, LinearDiscriminantAnalysis))

if __name__ == "__main__":
  unittest.main()
