import numpy as np
import pandas as pd

def generate_nonlin_cls(num_samples_grp = 100, r1 = 1, r2 = 0.7,
                        e = 0.2, seed = 1234):
  '''
  Generate a simple nonlinear classification data set

  Inputs:
  num_sample_grp: number of samples in each group
  r1: radius of first group
  r2: radius of second group
  e: random noise will be taken from a uniform distribution between -e and e
  seed: set random seed

  Returns:
  pandas dataframe
  '''
  np.random.seed(seed)
  theta = np.linspace(0, 2 * np.pi, num_samples_grp)
  a = r1 * np.cos(theta) + np.random.uniform(-e, e, num_samples_grp)
  b = r1 * np.sin(theta) + np.random.uniform(-e, e, num_samples_grp)
  c = r2 * np.cos(theta) + np.random.uniform(-e, e, num_samples_grp)
  d = r2 * np.sin(theta) + np.random.uniform(-e, e, num_samples_grp)
  out = pd.DataFrame({'x1' : np.append(a, c),
                      'x2' : np.append(b, d),
                      'outcome' : np.append([0] * num_samples_grp, [1] * num_samples_grp)})
  return out

def generate_lin_cls(num_samples_grp = 100, m1= -0.6, m2 = 0.6,
                     s1 = 1.0, s2 = 1.0, seed = 1234):
  '''
  Generate a simple linear classification data set

  Inputs:
  num_sample_grp: number of samples in each group
  m1: centroid of first group
  m2: centroiid of second group
  s1: variance of first group
  s2: variance of second group
  seed: set random seed

  Returns:
  pandas dataframe
  '''
  np.random.seed(seed)
  a = np.random.normal(m1, s1, num_samples_grp * 2)
  b = np.random.normal(m2, s2, num_samples_grp * 2)
  out = pd.DataFrame({'x1' : np.append(a[:num_samples_grp], b[:num_samples_grp]),
                      'x2' : np.append(a[num_samples_grp:], b[num_samples_grp:]),
                      'outcome' :  np.append([0] * num_samples_grp, [1] * num_samples_grp)
                       })
  return out

def generate_lin_reg(num_samples = 100, m = 0, s = 1.5, seed = 1234):
  '''
  Generate a simple linear regression data set

  Inputs:
  num_sample: number of samples
  m: mean for adding noise from normal distribution
  s: variance for adding noise from normal distribution
  seed: set random seed

  Returns:
  pandas dataframe
  '''
  np.random.seed(seed)
  x1 = np.linspace(0, 1, num_samples) * 2 + np.random.normal(m, s, num_samples)
  x2 = np.linspace(0, 1, num_samples) * 3 + np.random.normal(m, s, num_samples)
  y =  x1  +  x2
  out = pd.DataFrame({'x1' : x1, 'x2': x2,  'y' : y})
  return out


def generate_nonlin_reg(num_samples = 100, m1 = -1, s1 = 2.5, m2 =1, s2 = 2.5, seed = 1234):
  '''
  Generate a simple nonlinear regression data set

  Inputs:
  num_sample: number of samples
  m1: mean for adding noise from normal distribution
  s1: variance for adding noise from normal distribution
  m2: mean for adding noise from normal distribution
  s2: variance for adding noise from normal distribution
  seed: set random seed

  Returns:
  pandas dataframe
  '''
  np.random.seed(seed)
  x1 = np.linspace(0, 10, num_samples) ** 2 + np.random.normal(m1, s1, num_samples)
  x2 = np.linspace(0, 10, num_samples)  + np.random.normal(m2, s2, num_samples)
  y =  (x1  +  x2 * x1 + x1) / 50
  out = pd.DataFrame({'x1' : x1, 'x2': x2,  'y' : y})
  return out
