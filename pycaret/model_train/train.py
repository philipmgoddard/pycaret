import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import math
#import seaborn as sns

from pycaret.model_train.train_functions import resamp_loop, train_setup, update_summary
from pycaret.model_train.train_control import TrainControl
from pycaret.performance_metrics import defaultSummary

class Train():
  '''
  Objects of class Train are the main advantage of using the pycaret package.
  They are wrappers around the underlying model, but considerably quicker and
  easier to build.
  The underlying model is always available through the attribute fitted_model
  '''

  def __init__(self, x, y, method, preProcess = None,
               metric = None, trControl = None, tuneGrid = None,
               **kwargs):
    '''
    Train initialiser: build objects of class Train

    Inputs:
    x: pandas dataframe of the numerical features
    y: numpy array holding the outcome PG TODO: INVESTIGATE THIS
    method: string referenceing the underlying algorithm
    preProcess: object of class PreProcess. Not yet implemented
    metric: object from module pycaret.performance_metric.metric to evaluate the algorithm's performance]
    trControl: TrainControl object, specifying the training options
    tuneGrid: pandas dataframe specifying the hyperparameters to trial when training the model
    **kwargs: other arguments to pass. To pass to the underlying algorithm, use {'model_args': {'arg1': 123, 'arg2': 456,...}}
    '''

    #############################################
    # initialisation
    # TODO: add some getters/setters to ensure these set correctly
    #############################################

    self.train_input = x
    self.train_outcome = y
    self.method = method

    # TODO: put an exception here for multiclass classification
    # something for next version i thinks!

    model_args = kwargs.get('model_args', None)

    self.trControl, self.metric, self.tuneGrid, self.preProcess, summaryFunction, \
    metric_results, sumFunc_results, resamp_func, resamp_args, n_resamples, nrow_grid \
    = train_setup(trControl, method, metric, tuneGrid, preProcess)

    #############################################
    # model training
    #############################################

    # fit model with resampling, no hyperparams
    if self.tuneGrid is None:
      if model_args is None:
        model = self.method()
      else:
        model = self.method(**model_args)

      row_index = resamp_func(x.index.values, **resamp_args)

      # this can be parellelised with multiprocessing module
      metric_perf, perf = resamp_loop(self.train_input, self.train_outcome,
                                      row_index, n_resamples, model, self.metric,
                                      summaryFunction)


      # row is not defined! hard code to 0
      sumFunc_results = update_summary(0, perf, sumFunc_results, summaryFunction)
      metric_results.set_value(0, 'mean_' + self.metric.metric_name, metric_perf.mean())
      metric_results.set_value(0, 'sd_' + self.metric.metric_name, metric_perf.std())

    else: #HYPERPARAM SELECTION
      for row in range(nrow_grid):
        for hyp_param in self.tuneGrid: # is a pd dataframe

          if model_args:
            model_args.update({hyp_param : self.tuneGrid[hyp_param][row]})
          else:
            model_args = {hyp_param : self.tuneGrid[hyp_param][row]}

        model = self.method(**model_args)
        row_index = resamp_func(x.index.values, **resamp_args)

        metric_perf, perf = resamp_loop(self.train_input, self.train_outcome,
                                        row_index, n_resamples, model, self.metric,
                                        summaryFunction)

        metric_results.set_value(row, 'mean_' + self.metric.metric_name, metric_perf.mean())
        metric_results.set_value(row, 'sd_' + self.metric.metric_name, metric_perf.std())
        sumFunc_results = update_summary(row, perf, sumFunc_results, summaryFunction)

    #############################################
    # final selection
    #############################################

    best = self.metric().best(metric_results)
    self.best = best

    if self.tuneGrid is None:
      pass
    else:
      for hyp_param in self.tuneGrid:
        model_args.update({hyp_param : best[hyp_param].values[0]})
      model = self.method(**model_args)

    model.train(self.train_input, self.train_outcome)

    ##########################################
    # save predictions and optionally predprob as attribute
    # save metric and summary function results and model as attribute
    ##########################################
    self.pred_values = model.predict(self.train_input)
    if self.trControl.classProbs:
      self.pred_probs = model(self.train_input, predict_proba = True)

    self.metric_results = metric_results
    self.sumFunc_results = sumFunc_results
    self.fitted_model = model


  def predict(self, newdata, **kwargs):
    '''
    predict method accesses the predict method of the
    underlying model wrapper class.
    newdata: samples to make prediction for
    kwargs: arguments to be passed to the underlying predict method
    '''
    return self.fitted_model.predict(newdata, **kwargs)


  def plot(self, f_size = (6,6)):
    '''
    plot method allows quick visualistion of results over hyperparameter
    grid.
    For one hyperparameter, produces a simple x-y plot of hp1 vs result
    For two hyperparameters, produces an x-y plot of hp2 vs result for each value of hp1
    For three hyperparameters, produces a grid of x-y plots. Each facet is for a distinct hp1 value,
    displaying hp3 vs result for each value of hp2

    For >3 hyperparameters, the user should create their own plots.
    '''

    df = self.metric_results
    n_hyperparams = df.shape[1] - 2

    if n_hyperparams == 1:

      hp1 = df.columns.values[0]
      metric = df.columns.values[1]

      f, axarr = plt.subplots(1, 1, figsize = f_size, dpi=80)

      axarr.spines['right'].set_visible(False)
      axarr.spines['top'].set_visible(False)
      axarr.tick_params(axis=u'both', which=u'both',length=5)

      axarr.plot(df[hp1], df[metric], '-')
      axarr.scatter(df[hp1], df[metric])
      axarr.set_ylabel(metric)
      axarr.set_xlabel(hp1)

    elif n_hyperparams == 2:

      hp1 = df.columns.values[0]
      hp2 = df.columns.values[1]
      metric = df.columns.values[2]

      f, axarr = plt.subplots(1, 1, figsize = f_size, dpi=80)

      axarr.spines['right'].set_visible(False)
      axarr.spines['top'].set_visible(False)
      axarr.tick_params(axis=u'both', which=u'both',length=5)

      for _1 in df[hp1].drop_duplicates():

        hp2_tmp = df.loc[(df[hp1] == _1), [hp2]].values
        hp2_tmp = [x[0] for x in hp2_tmp]
        metric_tmp = df.loc[(df[hp1] == _1), [metric]].values
        metric_tmp = [x[0] for x in metric_tmp]

        axarr.plot(hp2_tmp, metric_tmp, '-x', label = _1)

      axarr.set_ylabel(metric)
      axarr.set_xlabel(hp2)
      axarr.legend(loc = 'best', title = hp1 )

    elif n_hyperparams == 3:
      # urg hacky

      hp1 = df.columns.values[0]
      hp2 = df.columns.values[1]
      hp3 = df.columns.values[2]
      metric = df.columns.values[3]

      # initialise grid
      n_hp1 = df[hp1].drop_duplicates().shape[0]
      n_row = math.ceil(n_hp1/ 3)
      n_col = 3

      f, axarr = plt.subplots(n_row, 3, figsize = f_size, dpi=80)

      ymin = min(df[metric].values) * 0.99
      ymax = max(df[metric].values) * 1.01

      i = 0
      j = 0
      plt_count = 0

      for _1 in df[hp1].drop_duplicates():
        for _2 in df[hp2].drop_duplicates():
          hp3_tmp = df.loc[(df[hp1] == _1), :].loc[(df[hp2] == _2), [hp3]].values
          hp3_tmp = [x[0] for x in hp3_tmp]
          metric_tmp = df.loc[(df[hp1] == _1), :].loc[(df[hp2] == _2), [metric]].values
          metric_tmp = [x[0] for x in metric_tmp]

          if n_row > 1:
            axarr[i][j].set_ylim([ymin, ymax])
            axarr[i][j].spines['right'].set_visible(False)
            axarr[i][j].spines['top'].set_visible(False)
            axarr[i][j].tick_params(axis=u'both', which=u'both',length=5)

            axarr[i][j].plot(hp3_tmp, metric_tmp, '-x', label = _2)
            axarr[i][j].set_title('{} = {}'.format(hp1, str(_1)))

            axarr[i][j].set_ylabel(metric)
            axarr[i][j].set_xlabel(hp3)

            if (i == 0) & (j == 2):
              axarr[i][j].legend(loc = 'best', title = hp2)

          else:
            axarr[j].set_ylim([ymin, ymax])
            axarr[j].spines['right'].set_visible(False)
            axarr[j].spines['top'].set_visible(False)
            axarr[j].tick_params(axis=u'both', which=u'both',length=5)

            axarr[j].plot(hp3_tmp, metric_tmp, '-x', label = _2)
            axarr[j].set_title('{} = {}'.format(hp1, str(_1)))

            axarr[j].set_ylabel(metric)
            axarr[j].set_xlabel(hp3)

            if j == 2:
              axarr[j].legend(loc = 'best', title = hp2)

            if j > 0:
              axarr[j].set_ylabel('')
              axarr[j].set_yticklabels([])
              axarr[j].set_yticks([])

        plt_count += 1
        j += 1
        if j == 3:
          i += 1
          j = 0

      # kill unused axis
      if n_row > 1:
        if n_row * 3 > len(df[hp2].drop_duplicates()):
          for k in range(j, 3):
            axarr[n_row-1][k].axis('off')

        # for axarr[i][j]

      f.subplots_adjust(hspace = 0.3)

    elif n_hyperparams > 3:
      raise ValueError('Too many hyperparameters for a simple plot. Go manual!')



