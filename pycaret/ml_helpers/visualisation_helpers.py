import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


def plot_pred_reg(pred, outcome):
  '''
  Plot prediction vs outcome for regression tasks.

  Inputs:
  pred: np array of predicted values
  outcome: np array of true values
  '''
  if len(pred) != len(outcome):
    raise ValueError('pred and outcome must be same length')
  sns.set_style('white')
  sns.despine()
  mn = min(pred.min(), outcome.min())
  mx = max(pred.max(), outcome.max())
  mx = mx + mx * 0.1
  mn = mn - mn * 0.1
  identity = np.arange(mn - 2 * mn, mx + 2 * mx)
  plt.scatter(pred, outcome, alpha = 0.6)
  plt.plot(identity, identity)
  plt.ylim(mn, mx)
  plt.xlim(mn, mx)
  plt.xlabel('prediction')
  plt.ylabel('outcome')
  plt.show()


def plot_roc(predprob, outcome):
  '''
  Plot roc curve. Uses sklearn.metrics

  Inputs:
  predprob: np array of predicted class probabilties for the positive class
  outcome: np array true classification
  '''
  fpr, tpr, _ = roc_curve(outcome, predprob)
  roc_auc = auc(fpr, tpr)

  sns.set_style('white')
  sns.despine()
  plt.plot(fpr, tpr, lw = 2, label='ROC curve (area = {:.3f})'.format(roc_auc))
  plt.plot([0, 1], [0, 1],color = 'black', lw = 2, linestyle='--')
  plt.xlim([-0.02, 1.02])
  plt.ylim([0.0, 1.02])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  plt.show()


