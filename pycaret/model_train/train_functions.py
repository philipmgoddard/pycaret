import numpy as np
import pandas as pd
import copy
from pycaret.performance_metrics.metrics import defaultSummary, Accuracy, RMSE
from pycaret.cross_val.resample import kfold, boot, repeated_cv
from pycaret.model_train.train_control import TrainControl


def train_setup(trControl = None, method = None, metric = None,
                tuneGrid = None, preProcess = None):
  '''
  Setup for TrainControl

  Inputs:
  trControl: object of class TrainControl
  method: class wrapping model object, containing __init__, train and predict methods
  metric: metric object for evaluating model performance
  tuneGrid: pandas dataframe of hyperparameters
  preProcess: currently not implemented
  '''

  ########################################
  #
  # Set defaults if arguments not specified
  #
  ########################################

  if not method:
    raise ValueError('Must specify ML algorithm')

  if not trControl:
      trControl = TrainControl()

  if not metric:
    if method.modelType == "Classification":
      metric = Accuracy
    elif method.modelType == "Regression":
      metric = RMSE
    else:
      raise NotImplementedError
  else:
    metric = metric

  if tuneGrid is None:
    tuneGrid = method.grid
  else:
   tuneGrid = tuneGrid

  summaryFunction = trControl.summaryFunction
  if summaryFunction == defaultSummary:
    summaryFunction = defaultSummary(method.modelType)

  sumFunc_results = copy.deepcopy(tuneGrid)
  metric_results = copy.deepcopy(tuneGrid)

  # will hit this for model with no hyperparameters
  if tuneGrid is None:
    nrow = 1
    sumFunc_results = pd.DataFrame()
    metric_results = pd.DataFrame()
  else:
    nrow = tuneGrid.shape[0]

  ########################################
  #
  # Set up pandas dataframe for to hold summary function and
  # metric results. Will hold:
  # | hyperparam1 | hyperparam2 | ... | hyperparam n | mean | sd |
  #
  ########################################
  for sumFunc in summaryFunction:
    sumFunc_results['mean_' + sumFunc.metric_name] = np.empty(nrow)
    sumFunc_results['sd_' + sumFunc.metric_name] = np.empty(nrow)

  metric_results['mean_' + metric.metric_name] = np.empty(nrow)
  metric_results['sd_' + metric.metric_name] = np.empty(nrow)

  ########################################
  #
  # Set up resampling method, and store options
  #
  ########################################
  if trControl.method == 'cv':
    resamp_func = kfold
    resamp_args = {'k' : trControl.number, 'seed' : trControl.seed}
    n_resamples = trControl.number
  elif trControl.method == 'boot':
    resamp_func = boot
    resamp_args ={'number' : trControl.number, 'seed' : trControl.seed}
    n_resamples = trControl.number
  elif trControl.method == 'repeated_cv':
    resamp_func = repeated_cv
    resamp_args ={'k' : trControl.number, 'seed' : trControl.seed,
                  'repeats' : trControl.repeats}
    n_resamples = trControl.number * trControl.repeats
  else:
    raise NotImplementedError

  # return options to Train
  return trControl, metric, tuneGrid, preProcess, summaryFunction, \
          metric_results, sumFunc_results, resamp_func, resamp_args, \
          n_resamples, nrow


def resamp_loop(train_input, train_outcome, preProcess, row_index,
               n_resamples, model, metric, summaryFunction):
  '''
  resample loop for hyperparameter selection.

  Inputs:
  train_input: pd dataframe holding features
  train_outcome: pd dataframe holding target.
  row_index: generator that yeilds index of pd dataframe for train/holdout.
             train_input and outcome need same index
  n_resamples: number of resamples for CV. eg for 10-fold CV this is 10
  metric: metric object used to evaluate best model
  summaryFunction: list holding metric objects for observing model performance

  returns: metric_perf and sumFunc_perf: np arrays holding mean and sd
           for metrics
  '''

  sumFunc_perf = np.empty([n_resamples, len(summaryFunction)])
  metric_perf = np.empty([n_resamples])
  index_values = train_input.index.values


  ########################################
  #
  # Loop over n_resamples. Get next array holding index to sample
  # for training and holdout. Train model on training set, evaluate
  # on holdout
  #
  ########################################
  for i in np.arange(n_resamples):
    train_samp = next(row_index)
    curr_train_input = train_input.iloc[train_samp]

    ########################################
    # Preprocessing if specified
    ########################################
    if preProcess is not None:
      for pp in preProcess:
        pp.fit(curr_train_input)
        curr_train_input = pp.transform(curr_train_input)

      # transform all input to ensure holdout is transformed in same way
      train_input = pp.transform(train_input)
    ########################################

    curr_train_outcome = train_outcome.iloc[train_samp]
    holdout = np.array([x for x in index_values if x not in train_samp])

    model.train(curr_train_input, curr_train_outcome)
    pred_tmp = model.predict(train_input.iloc[holdout])

    # need special case for AUC as require prob predictions to calculate
    # can probably optimise here for case if metric_perf in summmaryFunction
    if metric.metric_name == "auc":
      pred_proba_tmp = model.predict(train_input.iloc[holdout], predict_proba = True)[:, 1]
      metric_perf[i] = metric()(pred_proba_tmp, train_outcome.iloc[holdout].values)
    else:
      metric_perf[i] = metric()(pred_tmp, train_outcome.iloc[holdout].values)

    for j, sumFunc in enumerate(summaryFunction):
      if sumFunc.metric_name == "auc":
        pred_proba = model.predict(train_input.iloc[holdout], predict_proba = True)[:, 1]
        sumFunc_perf[i, j] = sumFunc()(pred_proba, train_outcome.iloc[holdout].values)
      else:
        sumFunc_perf[i, j] = sumFunc()(pred_tmp, train_outcome.iloc[holdout].values)

  return metric_perf, sumFunc_perf


def update_summary(row, perf, sumFunc_results, summaryFunction):
  '''
  Update summary function data frame
  '''
  for k in range(perf.shape[0]): # rows
    count = 0
    for l, sumFunc in enumerate(summaryFunction):
      sumFunc_results.set_value(row, 'mean_' + sumFunc.metric_name, perf[:, l].mean())
      sumFunc_results.set_value(row, 'sd_' + sumFunc.metric_name, perf[:, l].std())

  return sumFunc_results
