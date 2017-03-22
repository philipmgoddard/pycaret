import numpy as np
from pycaret.performance_metrics import defaultSummary

class TrainControl():
  '''
  Define the control object
  Inspired by Max Kuhn's R caret package
  '''
  allowed_methods = ['cv', 'repeated_cv', 'boot']

  def __init__(self, method = 'cv', number = None,
               repeats = None, classProbs = None,
               summaryFunction = None, seed = None):
    '''
    Initialise TrainControl object

    Inputs:
    method: string referencing thecross validation scheme. allowed values are 'cv', 'repeatedcv', 'boot'
    number: number of folds (cv) or bootstrap resamples
    repeats: number of repeats for repeated cv
    classProbs: if a classification model, retain class probabilities
    summaryFunction: list of summary function objects (see pycaret.performance_metrics.metrics)
    seed: random seed for sampling
    '''
    self.method = method
    self.number = number
    self.repeats = repeats
    self.classProbs = classProbs
    self.summaryFunction = summaryFunction
    self.seed = seed

    ##############################################
    # error checking and defaults for TrainControl
    # options below
    ##############################################

  @property
  def method(self):
    return self._method

  @method.setter
  def method(self, value):
    if value not in TrainControl.allowed_methods:
      raise NotImplementedError
    self._method = value

  @property
  def number(self):
    return self._number

  @number.setter
  def number(self, value):
    if not value:
      if self.method == 'cv':
        self._number = 10
      else:
        self._number = 25
    else:
      if not isinstance(value, int):
        raise ValueError
      else:
        self._number = value

  @property
  def repeats(self):
    return self._repeats

  @repeats.setter
  def repeats(self, value):
    if not value:
      if self.method == 'cv':
        self._repeats = 1
      else:
        self._repeats = self.number
    else:
      if not isinstance(value, int):
        raise ValueError
      else:
        self._repeats = value

  @property
  def classProbs(self):
    return self._classProbs

  @classProbs.setter
  def classProbs(self, value):
    if not value:
      self._classProbs = False
    else:
      if isinstance(value, bool):
        self._classProbs = value
      else:
        raise ValueError

  @property
  def summaryFunction(self):
    return self._summaryFunction

  @summaryFunction.setter
  def summaryFunction(self, sum_func):
    if sum_func is None:
      self._summaryFunction = defaultSummary
    else:
      self._summaryFunction = sum_func

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, value):
    if not value:
      self._seed = 1234
    else:
      if isinstance(value, int):
        self._seed = value
      else:
        raise ValueError


  def __str__(self):
    '''
    Basic string representation for TrainControl objects.
    '''
    return """Train Control Options:\n
              \tMethod: {}
              \tNumber: {}
              \tRepeats: {}
          """.format(self.method, self.number, self.repeats)


