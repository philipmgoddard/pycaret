import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score


class Accuracy():
  '''
  Accuracy: proportion of predicted = target
  '''
  metric_name = "acc"

  def __call__(self, pred, outcome):
    return np.sum(pred == outcome) / pred.size

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_acc', ascending = False).head(1)


class AUC():
  '''
  Area under ROC curve. Wrapper around sklearn roc_auc_score.
  Requires probability predictions and class outcomes.
  '''
  metric_name = "auc"

  def __call__(self, pred, outcome):
    return roc_auc_score(outcome, pred)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_auc', ascending = False).head(1)


class FN():
  '''
  False negatives: number of predict negative class and true outcome positive class
  '''
  metric_name = "FN"

  def __call__(self, pred, outcome):
    return np.sum((pred == 0) & (outcome == 1))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_FN', ascending = True).head(1)


class FP():
  '''
  False positives: number of predict positive class and true outcome negative class
  '''
  metric_name = "FP"

  def __call__(self, pred, outcome):
    return np.sum((pred == 1) & (outcome == 0))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_FN', ascending = True).head(1)


class TP():
  '''
  True positives: number of predict positive class and outcome positive class
  '''
  metric_name = "TP"

  def __call__(self, pred, outcome):
    return np.sum((pred[outcome == 1]) == (outcome[outcome == 1]))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_TP', ascending = False).head(1)


class TN():
  '''
  True negatives: number of predict negative class and outcome negative class
  '''
  metric_name = "TN"

  def __call__(self, pred, outcome):
    return np.sum((pred[outcome == 0]) == (outcome[outcome == 0]))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_TN', ascending = False).head(1)


class Sensitivity():
  '''
  Sensitivity: equivilent to recall.
  '''
  metric_name = "sens"

  def __call__(self, pred, outcome):
    tp = TP()(pred, outcome)
    fn = FN()(pred, outcome)
    return tp / (tp + fn)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_sens', ascending = False).head(1)


class Specificity():
  '''
  Specificity
  '''
  metric_name = "spec"

  def __call__(self, pred, outcome):
    tn = TN()(pred, outcome)
    fp = FP()(pred, outcome)
    return tn / (tn + fp)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_spec', ascending = False).head(1)


class Precision():
  '''
  Precision
  '''
  metric_name = "prec"

  def __call__(self, pred, outcome):
    tp = TP()(pred, outcome)
    fp = FP()(pred, outcome)
    return tp / (tp + fp)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_prec', ascending = False).head(1)


class F1():
  '''
  F1 score:
  '''
  metric_name = "F1"

  def __call__(self, pred, outcome):
    tp = TP()(pred, outcome)
    fn = FN()(pred, outcome)
    fp = FP()(pred, outcome)
    return 2.0 * tp / (2.0 * tp + fp + fn)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_F1', ascending = False).head(1)

class Kappa():
  '''
  Cohen's kappa statistic
  '''
  metric_name = "kappa"

  def __call__(self, pred, outcome):
    tp = TP()(pred, outcome)
    tn = TN()(pred, outcome)
    fp = FP()(pred, outcome)
    fn = FP()(pred, outcome)
    n_pred = len(pred)

    p_0 = (tp + tn) / n_pred
    marginal_a = ((tp + fp) * (tp + fn)) / n_pred
    marginal_b = ((tn + fn) * (tn + fp)) / n_pred
    p_e = (marginal_a + marginal_b) / n_pred

    return 1.0 - ((1.0 - p_0) / (1.0 - p_e))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_kappa', ascending = False).head(1)

class RMSE():
  '''
  Root mean square error
  '''
  metric_name = "RMSE"

  def __call__(self, pred, outcome):
    return np.sqrt(np.mean(np.power(pred - outcome, 2)))

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_RMSE', ascending = True).head(1)


class R2():
  '''
  R2 aka coefficient of determination
  Use sklearn r2_score function
  '''
  metric_name = "R2"

  def __call__(self, pred, outcome):
    return r2_score(outcome, pred, sample_weight = None, multioutput = None)

  @staticmethod
  def best(results):
    return results.sort_values(by = 'mean_R2', ascending = False).head(1)


def defaultSummary(modelType = None):
  '''
  Default summary for classification or regression models
  '''
  if modelType is None:
    raise ValueError("specify Classification or Regression")
  if modelType == "Classification":
    return [Accuracy, Kappa, F1]
  elif modelType == "Regression":
    return [RMSE, R2]
  else:
    raise NotImplementedError

def twoClassSummary(modelType = None):
  '''
  A useful summary for binary classificcation models
  '''
  if modelType == "Classification":
    return [AUC, Sensitivity, Specificity]
  else:
    raise ValueError
