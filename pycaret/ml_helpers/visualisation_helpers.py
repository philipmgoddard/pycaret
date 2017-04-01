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



def kde_plot(df, n_col = 3, outcome_col = None, plot_legend = False, cov_factor = 0.25, f_size = (15,15)):
  from scipy.stats import gaussian_kde

  if outcome_col is None:
    n_features = df.shape[1]
    feature_names = df.columns.values
    df_features = df
  else:
    n_features = df.shape[1] -1
    feature_names = [x for x in df.columns.values if x != outcome_col]
    df_features = df.loc[:, feature_names]
    outcome_values = sorted([x for x in blah.loc[:, outcome_col].unique()])

  if n_features % n_col == 0:
    n_row = n_features // n_col
  else:
    n_row = n_features // n_col + 1

  f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

  v = 0
  for i in np.arange(n_row):
    for j in np.arange(n_col):

      xmin = df_features.loc[:, feature_names[v]].min()
      xmax = df_features.loc[:, feature_names[v]].max()
      axarr[i][j].spines['right'].set_visible(False)
      axarr[i][j].spines['top'].set_visible(False)
      axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

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

      if v == n_features: break

  f.subplots_adjust(hspace = 0.5)
  #return f


def hist_plot(df, n_col = 3, outcome_col = None, plot_legend = False,
              norm = True, f_size = (15,15)):

  if outcome_col is None:
    n_features = df.shape[1]
    feature_names = df.columns.values
    df_features = df
  else:
    n_features = df.shape[1] -1
    feature_names = [x for x in df.columns.values if x != outcome_col]
    df_features = df.loc[:, feature_names]
    outcome_values = sorted([x for x in blah.loc[:, outcome_col].unique()])

  if n_features % n_col == 0:
    n_row = n_features // n_col
  else:
    n_row = n_features // n_col + 1

  f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

  v = 0
  for i in np.arange(n_row):
    for j in np.arange(n_col):
      xmin = df_features.loc[:, feature_names[v]].min()
      xmax = df_features.loc[:, feature_names[v]].max()
      axarr[i][j].spines['right'].set_visible(False)
      axarr[i][j].spines['top'].set_visible(False)
      axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

      if outcome_col is not None:
        for c in outcome_values:
          lab = None if plot_legend is False else c
          if n_row == 1:
            axarr[j].hist(df_features.loc[df[outcome_col] == c,
                          feature_names[v]], label = lab,
                          normed = norm, alpha = 0.6)
            axarr[j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[j].legend()
          else:
            axarr[i,j].hist(df_features.loc[df[outcome_col] == c,
                            feature_names[v]], label = lab,
                            normed = norm, alpha = 0.6)
            axarr[i,j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[i,j].legend()
          v += 1
        else:
          if n_row == 1:
            axarr[j].hist(df_features.loc[ : , feature_names[v]],
                          normed = norm, alpha = 0.6)
            axarr[j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[j].legend()
          else:
            axarr[i,j].hist(df_features.loc[ : , feature_names[v]],
                            normed = norm, alpha = 0.6)
            axarr[i,j].set_title(feature_names[v])
            if plot_legend:
              if (j == 0) & (i == 0):
                axarr[i,j].legend()
          v += 1

        if v == n_features: break

  f.subplots_adjust(hspace = 0.5)
  #return f



def cs(var):
  return (var - var.mean()) / var.max()


def pairwise_plot(df, outcome_col = None, center_scale = True,
                  plot_legend = False, f_size = (15,15)):

  if outcome_col is None:
    n_features = df.shape[1]
    feature_names = df.columns.values
    df_features = df
  else:
    n_features = df.shape[1] -1
    feature_names = [x for x in df.columns.values if x != outcome_col]
    df_features = df.loc[:, feature_names]
    outcome_values = sorted([x for x in blah.loc[:, outcome_col].unique()])

  if center_scale:
    df_features = df_features.apply(lambda x: cs(x))

  n_features = df_features.shape[1]
  if n_features == 1:
    raise NotImplementedError('not implemented for n_features = 1')

  f, axarr = plt.subplots(n_features, n_features, figsize = f_size, dpi=80)

  v = 0
  for i in range(n_features):
    for j in range(n_features):
      axarr[i][j].spines['right'].set_visible(False)
      axarr[i][j].spines['top'].set_visible(False)

      if outcome_col is not None:
        for c in outcome_values:
          tmp_i = cs(inputTrain.loc[:, numVar[i]])
          tmp_j = cs(inputTrain.loc[:, numVar[j]])
          if j <= i:
            axarr[i,j].scatter(tmp_i[df[outcome_col] == c],
                               tmp_j[df[outcome_col] == c],
                               label = lab,
                               alpha = 0.5,
                               s = 2)

            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_xticklabels([])

            if j == 0:
              axarr[i,j].set_ylabel(numVar[i], rotation = 45)

            if i== nVar - 1:
              axarr[i,j].set_xlabel(numVar[j], rotation = 45)
            else:
              axarr[i,j].axis('off')
        else:
          tmp_i = cs(inputTrain.loc[:, numVar[i]])
          tmp_j = cs(inputTrain.loc[:, numVar[j]])
          if j <= i:
            axarr[i,j].scatter(tmp_i,
                               tmp_j,
                               label = lab,
                               alpha = 0.5,
                               s = 2)

            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_xticklabels([])

            if j == 0:
              axarr[i,j].set_ylabel(numVar[i], rotation = 45)

            if i== nVar - 1:
              axarr[i,j].set_xlabel(numVar[j], rotation = 45)
          else:
            axarr[i,j].axis('off')

  f.subplots_adjust(hspace = 0.5)
  f.suptitle('(centered and scaled) pairwise plot', fontsize = 24)
  f.subplots_adjust(top=0.95)
  #return f



def cat_plot(df, outcome_col = None,
             n_col = 3, plot_legend = False,
             f_size = (15,15)):

  if outcome_col is None:
    raise ValueError('outcome column cannot be None')
  else:
    n_features = df.shape[1] -1
    feature_names = [x for x in df.columns.values if x != outcome_col]
    outcome_values = df.loc[:, outcome_col].unique()

  if n_features % n_col == 0:
    n_row = n_features // n_col
  else:
    n_row = n_features // n_col + 1

  f, axarr = plt.subplots(n_row, n_col, figsize = f_size, dpi=80)

  v = 0
  for i in range(n_row):
    for j in range (n_col):

      axarr[i][j].spines['right'].set_visible(False)
      axarr[i][j].spines['top'].set_visible(False)
      axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

      if v >= n_features:
        axarr[i,j].axis('off')
      else:
        n_levels = len(df.loc[:, feature_names[v]].unique())
        width = 0.45
        ind = np.arange(n_levels)

        tmp = (df
        .loc[:, [feature_names[v], outcome_col]]
        .groupby([feature_names[v], outcome_col])[feature_names[v]]
        .count()
        .unstack(outcome_col)
        .fillna(0)
        .loc[:, outcome_values] )

        for d in range(len(outcome_values)):
          if d == 0:
            d0 = tmp[outcome_values[d]].values
            axarr[i,j].bar(ind, d0, width, alpha = 0.8, label = outcome_values[d])
          else:
            dd = tmp[outcome_values[d]].values
            axarr[i,j].bar(ind, dd, width, bottom = d0, alpha = 0.8, label = outcome_values[d])

        if n_row == 1:
          axarr[j].set_xticks(ind)
          axarr[j].set_xticklabels(tmp.index.values.tolist(), rotation = 45)
          axarr[j].set_title(feature_names[v])
          if plot_legend:
            if (j == 0) & (i == 0):
              axarr[j].legend()
        else:
          axarr[i,j].set_xticks(ind)
          axarr[i,j].set_xticklabels(tmp.index.values.tolist(), rotation = 45)
          axarr[i,j].set_title(feature_names[v])

          if plot_legend:
            if (j == 0) & (i == 0):
              axarr[i,j].legend()
      v += 1

  f.subplots_adjust(hspace = 0.5)
  #return f
