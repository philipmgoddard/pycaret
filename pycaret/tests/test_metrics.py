import unittest
import numpy as np
import pycaret.performance_metrics.metrics as metrics

class MetricTestCase(unittest.TestCase):
  '''
  tests for performance metrics
  '''

  pred =     np.array([1, 0, 1, 0, 1])
  predprob = np.array([0.51, 0.3, 0.9, 0.1, 0.7])
  outcome =  np.array([1, 0, 0, 1, 1])

  predr =    np.array([0.6, 0.3, 0.8, 0.2, 0.1])
  outcomer = np.array([0.55, 0.2, 0.83, 0.12, 0.2])

  def test_Accuracy(self):
    self.assertEqual(metrics.Accuracy()
      (MetricTestCase.pred, MetricTestCase.outcome), 0.6)

  def test_AUC(self):
    self.assertAlmostEqual(metrics.AUC()
        (MetricTestCase.predprob, MetricTestCase.outcome), 0.3333, places = 3)

  def test_TP(self):
    self.assertEqual(metrics.TP()
      (MetricTestCase.pred, MetricTestCase.outcome), 2)

  def test_TN(self):
    self.assertEqual(metrics.TN()
      (MetricTestCase.pred, MetricTestCase.outcome), 1)

  def test_FP(self):
    self.assertEqual(metrics.FP()
      (MetricTestCase.pred, MetricTestCase.outcome), 1)

  def test_FN(self):
    self.assertEqual(metrics.FN()
      (MetricTestCase.pred, MetricTestCase.outcome), 1)

  def test_Sensitivity(self):
    self.assertAlmostEqual(metrics.Sensitivity()
    (MetricTestCase.pred, MetricTestCase.outcome), 0.6666, places = 3)

  def test_Specificity(self):
    self.assertAlmostEqual(metrics.Specificity()
    (MetricTestCase.pred, MetricTestCase.outcome), 0.5, places = 3)

  def test_Precision(self):
    self.assertAlmostEqual(metrics.Precision()
    (MetricTestCase.pred, MetricTestCase.outcome), 0.6666, places = 3)

  def test_F1(self):
    self.assertAlmostEqual(metrics.F1()
      (MetricTestCase.pred, MetricTestCase.outcome), 0.6666, places = 3)

  def test_Kappa(self):
    self.assertAlmostEqual(metrics.Kappa()
      (MetricTestCase.pred, MetricTestCase.outcome), 0.1666, places = 3)

  def test_RMSE(self):
    self.assertAlmostEqual(metrics.RMSE()
      (MetricTestCase.predr, MetricTestCase.outcomer), 0.07720, places = 4)

  # in caret, this is 'traditional' formula
  def test_R2(self):
    self.assertAlmostEqual(metrics.R2()
      (MetricTestCase.predr, MetricTestCase.outcomer), 0.91808, places = 4)


class DefaultSummaryTestCase(unittest.TestCase):
  ''' ensure that when defaultSummary instantiated gives correct behaviour'''

  def test_cls_case(self):
    self.assertEqual(metrics.defaultSummary("Classification"),
      [metrics.Accuracy, metrics.Kappa, metrics.F1])

  def test_reg_case(self):
    self.assertEqual(metrics.defaultSummary("Regression"),
      [metrics.RMSE, metrics.R2])

  def test_argument_none(self):
    self.assertRaises(ValueError, metrics.defaultSummary, None)

  def test_argument_wrong(self):
    self.assertRaises(NotImplementedError, metrics.defaultSummary, "TimeSeries")


class TwoClassSummaryTestCase(unittest.TestCase):
  ''' ensure that twoClassSummary instantiated gives correct behaviour'''

  def test_cls_case(self):
    self.assertEqual(metrics.twoClassSummary("Classification"),
      [metrics.AUC, metrics.Sensitivity, metrics.Specificity])

  def test_argument_wrong(self):
    self.assertRaises(ValueError, metrics.twoClassSummary, None)


if __name__ == "__main__":
  unittest.main()

