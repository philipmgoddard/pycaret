import unittest
import pandas as pd
from pycaret.utils import sampling as samp
from pycaret.utils.helpers import expand_grid
from pycaret.utils import data_gen as gen

class DataGenTestCase(unittest.TestCase):
  '''
  tests for data generation functions
  '''
  def setUp(self):
    self.nlc = gen.generate_nonlin_cls(num_samples_grp = 100)
    self.lc = gen.generate_lin_cls(num_samples_grp = 100)
    self.nlr = gen.generate_nonlin_reg(num_samples = 100)
    self.lr = gen.generate_nonlin_reg(num_samples = 100)

  def test_generate_nonlin_cls(self):
    self.assertTrue(isinstance(self.nlc, pd.DataFrame))
    self.assertTrue(self.nlc.shape[0] == 200)
    self.assertTrue(self.nlc.query('outcome == 0').shape[0] == 100)

  def test_generate_nonlin_cls(self):
    self.assertTrue(isinstance(self.lc, pd.DataFrame))
    self.assertTrue(self.lc.shape[0] == 200)
    self.assertTrue(self.lc.query('outcome == 0').shape[0] == 100)

  def test_generate_nonlin_reg(self):
    self.assertTrue(isinstance(self.nlr, pd.DataFrame))
    self.assertTrue(self.nlr.shape[0] == 100)

  def generate_lin_reg(self):
    self.assertTrue(isinstance(self.lr, pd.DataFrame))
    self.assertTrue(self.lr.shape[0] == 100)


class HelpersTestCase(unittest.TestCase):
  '''
  tests for grid expansion function
  '''
  def test_expand_grid(self):
    tmp1 = expand_grid({'a': [1,2], 'b':[3, 4]})
    tmp2 = pd.DataFrame({'a':[1,1,2,2], 'b':[3,4,3,4]})
    self.assertTrue((tmp1 == tmp2).all().all())


class SamplingTestCase(unittest.TestCase):
  '''
  tests for up and down sampling
  '''
  def setUp(self):
    self.df = pd.DataFrame({'var1' :    [1,2,3,4,5,6],
                            'var2' :    [4,6,2,3,4,2],
                            'outcome' : [1,1,0,0,0,0]})

  def test_downsample(self):
    '''
    test downsampling function
    '''
    tmp = samp.downSample(X = self.df,  y ='outcome', seed = 1234)

    self.assertTrue(tmp.shape[0] == 4)
    self.assertTrue(tmp.loc[tmp.outcome==1, :].shape[0] == 2)
    self.assertTrue(tmp.loc[tmp.outcome==0, :].shape[0] == 2)

  def test_upsample(self):
    '''
    test downsampling function
    '''
    tmp = samp.upSample(X = self.df,  y ='outcome', seed = 1234)

    self.assertTrue(tmp.shape[0] == 8)
    self.assertTrue(tmp.loc[tmp.outcome==1, :].shape[0] == 4)
    self.assertTrue(tmp.loc[tmp.outcome==0, :].shape[0] == 4)


if __name__ == "__main__":
  unittest.main()
