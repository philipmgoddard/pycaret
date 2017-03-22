import unittest
import numpy as np
import copy
import pycaret.cross_val.resample as resample


class ResampleTestCase(unittest.TestCase):
  '''
  Tests for resampling schemes
  '''

  index = np.arange(10)

  def test_kfold(self):
    nfold = 4
    kfold_generator = resample.kfold(ResampleTestCase.index, k = nfold, seed = 1234)
    folds = [next(kfold_generator) for i in np.arange(nfold)]

    self.assertEqual(len(folds), 4)
    self.assertRaises(StopIteration, kfold_generator.__next__)


  def test_repeatedCV(self):

    nfold = 4
    nrepeats = 3
    repeatcv_generator = resample.repeated_cv(ResampleTestCase.index,
                                              repeats = nrepeats,
                                              k = nfold,
                                              seed = 1234)

    # need to copy
    folds = [copy.copy(next(repeatcv_generator))
            for i in np.arange(nfold * nrepeats) ]

    self.assertEqual(len(folds), 12)
    self.assertNotEqual([x for x in folds[0]], [x for x in folds[4]])
    self.assertRaises(StopIteration, repeatcv_generator.__next__)


  def test_boot(self):

    number = 10
    boot_generator = resample.boot(ResampleTestCase.index,
                                              number = number,
                                              seed = 1234)

    # need to copy
    resamps = [copy.copy(next(boot_generator))
              for i in np.arange(number) ]

    self.assertEqual(len(resamps), 10)
    self.assertNotEqual([x for x in resamps[0]], [x for x in resamps[1]])
    self.assertRaises(StopIteration, boot_generator.__next__)


if __name__ == '__main__':
  main()
