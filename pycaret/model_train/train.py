import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

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


  # TESTS THIS!!!
  def predict(self, newdata, **kwargs):
    '''
    Useful docstring
    '''
    return self.fitted_model.predict(newdata, **kwargs)


  def plot(self):
    '''
    useful docstring
    '''
    df = self.metric_results
    n_hyperparams = df.shape[1] - 2
    sns.set_style('white')

    if n_hyperparams == 1:
      hp1 = df.columns.values[0]
      metric = df.columns.values[1]
      sns.factorplot(x = hp1, y = metric, data = df, facet_kws={'size' : 5})
      plt.show()
    elif n_hyperparams == 2:
      hp1 = df.columns.values[0]
      hp2 = df.columns.values[1]
      metric = df.columns.values[2]
      sns.factorplot(x = hp1, y = metric, hue = hp2, data = df, facet_kws={'size' : 5})
      plt.show()
    elif n_hyperparams == 3:
      hp1 = df.columns.values[0]
      hp2 = df.columns.values[1]
      hp3 = df.columns.values[2]
      metric = df.columns.values[3]
      sns.factorplot(x = hp1, y = metric, hue = hp2, col = hp3, data = df, facet_kws={'size' : 5})
      plt.show()
    elif n_hyperparams > 3:
      raise ValueError('Too many hyperparameters for an intuitive plot. Go manual!')



