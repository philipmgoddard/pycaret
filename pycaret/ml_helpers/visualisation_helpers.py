import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy.stats import gaussian_kde



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


def kde_plot(df, n_col = 3, outcome_col = None,
             plot_legend = False, cov_factor = 0.25,
             f_size = (15, 15)):

  if outcome_col is None:
    n_features = df.shape[1]
    feature_names = df.columns.values
    df_features = df
  else:
    n_features = df.shape[1] -1
    feature_names = [x for x in df.columns.values if x != outcome_col]
    df_features = df.loc[:, feature_names]
    outcome_values = df.loc[:, outcome_col].unique()

  if n_features % ncol == 0:
    n_row = n_features // n_col
  else:
    n_row = n_features // n_col + 1

  f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

  v = 0
  for i in np.arange(n_row):
    for j in np.arange(n_col):
      xmin = df_features.loc[:, feature_names[v]].min()
      xmax = df_features.loc[:, feature_names[v]].max()
      if outcome_col is not None:
        for c in outcome_values:
          lab = None if plot_legend is False else c
          density = gaussian_kde(df_features.loc[df[outcome_col] == c , feature_names[v]])
          density.covariance_factor = lambda : cov_factor
          density._compute_covariance()
          xs = np.arange(xmin, xmax, 0.1)
          if n_row == 1:
            axarr[j].plot(xs, density(xs), label = lab)
            axarr[j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[j].legend()
          else:
            axarr[i,j].plot(xs, density(xs), label = lab)
            axarr[i,j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[i,j].legend()
        v += 1
      else:
        density = gaussian_kde(df_features.loc[ : , feature_names[v]])
        density.covariance_factor = lambda : cov_factor
        density._compute_covariance()
        xs = np.arange(xmin, xmax, 0.1)
        axarr[i,j].plot(xs, density(xs))
        axarr[i,j].set_title(feature_names[v])
        v += 1
  f.subplots_adjust(hspace = 0.5)
  #return f
