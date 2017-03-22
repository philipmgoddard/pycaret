import unittest
import pandas as pd
import pycaret.model_train.train_control as train_control
import pycaret.model_train.train_functions as train_func
from pycaret.performance_metrics.metrics import defaultSummary, AUC, Accuracy
from pycaret.performance_metrics.metrics import Kappa, F1, RMSE, R2
from pycaret.cls_models.linear_models import GLM_L1
from pycaret.reg_models.linear_models import LM
from pycaret.cross_val.resample import kfold, boot

class TrainControlDefaultTestCase(unittest.TestCase):
  '''
  test default TrainControl object
  '''

  def setUp(self):
    self.trControl = train_control.TrainControl()

  def test_method(self):
    self.assertTrue(self.trControl.method == 'cv')

  def test_number(self):
    self.assertTrue(self.trControl.number == 10)

  def test_number(self):
    self.assertTrue(self.trControl.repeats == 1)

  def test_classProbs(self):
    self.assertFalse(self.trControl.classProbs)

  def test_summaryFunction(self):
    self.assertTrue(self.trControl.summaryFunction == defaultSummary)

  def test_seed(self):
    self.assertEqual(self.trControl.seed, 1234)


class TrainControlTestCase(unittest.TestCase):
  '''
  Test user defined TrainControl object
  '''
  def setUp(self):
    self.trControl = train_control.TrainControl(
                                    method = 'boot',
                                    number = 20,
                                    repeats = None,
                                    classProbs = True,
                                    summaryFunction = [AUC],
                                    seed = 432)

  def test_method(self):
    self.assertTrue(self.trControl.method == 'boot')

  def test_number(self):
    self.assertTrue(self.trControl.number == 20)

  def test_number(self):
    self.assertTrue(self.trControl.repeats == 20)

  def test_classProbs(self):
    self.assertTrue(self.trControl.classProbs)

  def test_summaryFunction(self):
    self.assertTrue(self.trControl.summaryFunction == [AUC])

  def test_seed(self):
    self.assertEqual(self.trControl.seed, 432)


class TrainFunctionsTestCase(unittest.TestCase):
  '''
  Test model_train.train_functions
  '''

  def test_train_setup_default(self):
    tmp = train_func.train_setup(method = GLM_L1)

    # default metric is accuracy for cls model
    self.assertTrue(tmp[1] == Accuracy)
    # tuneGrid is pd.DataFrame
    self.assertTrue(isinstance(tmp[2], pd.DataFrame))
    # preProcess is None
    self.assertTrue(tmp[3] is None)
    # default summaryFunction
    self.assertTrue(tmp[4] == [Accuracy, Kappa, F1])
    # metric results is pandas dataframe
    self.assertTrue(isinstance(tmp[5], pd.DataFrame))
    # metric results has same number of rows as grid
    self.assertTrue(tmp[5].shape[0] == tmp[2].shape[0])
    # sumFunc_results df
    self.assertTrue(tmp[6].shape[0] == tmp[2].shape[0])
    self.assertTrue(tmp[6].shape[1] == len(tmp[4]) * 2 + 1)
    # resamp_func
    self.assertTrue(tmp[7] == kfold)
    self.assertTrue(tmp[8]['k'] == 10)
    self.assertTrue(tmp[8]['seed'] == 1234)
    # n resamples
    self.assertTrue(tmp[9] == 10)
    # nrow
    self.assertTrue(tmp[10] == tmp[6].shape[0])


  def test_train_setup(self):
    '''
    Test train setup function
    '''
    tmp_trControl = train_control.TrainControl(method = 'boot', number = 20,
                                 repeats = None, classProbs = None,
                                 summaryFunction = defaultSummary, seed = 5678)
    tmp = train_func.train_setup(trControl = tmp_trControl, method = LM)

    # default metric is accuracy for cls model
    self.assertTrue(tmp[1] == RMSE)
    # tuneGrid is pd.DataFrame, or None
    self.assertTrue(tmp[2] is None)
    # preProcess is None
    self.assertTrue(tmp[3] is None)
    # default summaryFunction
    self.assertTrue(tmp[4] == [RMSE, R2])
    # metric results is pandas dataframe
    self.assertTrue(isinstance(tmp[5], pd.DataFrame))
    # metric results has same number of rows as grid
    self.assertTrue(tmp[5].shape[0] == 1)
    # sumFunc_results df
    self.assertTrue(tmp[6].shape[0] == 1)
    self.assertTrue(tmp[6].shape[1] == len(tmp[4]) * 2)
    # resamp_func
    self.assertTrue(tmp[7] == boot)
    self.assertTrue(tmp[8]['number'] == tmp_trControl.number)
    self.assertTrue(tmp[8]['seed'] == tmp_trControl.seed)
    # n resamples
    self.assertTrue(tmp[9] == tmp_trControl.number)
    # nrow
    self.assertTrue(tmp[10] == 1)


if __name__ == "__main__":
  unittest.main()
