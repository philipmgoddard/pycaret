import unittest
import pandas as pd
import pycaret.ml_helpers.feature_selection as fs

class FeatureSelectionTestCase(unittest.TestCase):
  '''
  tests for ml_helpers.feature_selection module
  '''

  def setUp(self):
    self.df = pd.DataFrame({'var1' : [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                            'var2' : [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                            'var3' : [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]})

    self.corr_df = pd.DataFrame({'var1': [2,0,0,6,9,2],
                                 'var2': [1,2,3,4,5,6],
                                 'var3': [2,4,6,8,10,12],
                                 'var4': [2,4,5,0,10,12]})


  def test_nearZeroVariance(self):
    '''
    test for nearZeroVariance
    '''
    tmp = fs.nearZeroVariance(self.df)
    self.assertTrue(tmp[0][0] == 'var1')
    self.assertTrue(len(tmp[1]) == 2)
    self.assertTrue('var1' in tmp[1])
    self.assertTrue('var2' in tmp[1])

  def test_findCorrelation(self):
    '''
    test for findCorrelation
    '''

    tmp = fs.findCorrelation(self.corr_df, threshold = 0.9)
    self.assertTrue(tmp[0] == 'var2')
    self.assertTrue(len(tmp) == 1)


if __name__ == "__main__":
  unittest.main()
