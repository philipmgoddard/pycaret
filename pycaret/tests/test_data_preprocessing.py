import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pycaret.data_preprocessing.pp import CS_Process, PCA_Process



class DataPreprocessTestCase(unittest.TestCase):
  '''
  basic tests of pp module class
  '''

  def test_CS_Preprocess(self):
    tmp_CS = CS_Process()
    self.assertTrue(isinstance(tmp_CS.pp_method, StandardScaler))

  def test_PCA_Preprocess(self):
    tmp_PCA1 = PCA_Process()
    self.assertTrue(isinstance(tmp_PCA1.pp_method, PCA))


if __name__ == "__main__":
  unittest.main()
