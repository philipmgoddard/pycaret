# pycaret

A Python Classification and Regression training framework.

Heavily inspired by Max Kuhn's legendary R caret package https://topepo.github.io/caret/



### An example:

``` python
import pandas as pd
from pycaret import Train, TrainControl, expand_grid
from pycaret.performance_metrics.metrics import defaultSummary, Accuracy
from pycaret.cls_models.trees_rules import RF_Cls
from pycaret.utils.data_gen import generate_nonlin_cls

dat = generate_lin_cls()
inputTrain = dat.loc[:, ['x1', 'x2']]
outcomeTrain = dat.outcome

trControl = TrainControl(method = "cv",
                       number = 5,
                       summaryFunction = defaultSummary('Classification'))

rf_model = RF_Cls
grid_cust = expand_grid({'n_estimators' : [50, 100, 250],
                     'max_depth' : [3, 6, 9],
                     'max_features' : [1, 2]})

rfTrain = Train(inputTrain,
              outcomeTrain,
              rf_model,
              metric = Accuracy,
              trControl = trControl,
              tuneGrid = grid_cust)

rfTrain.plot()
```

### Purpose

pycaret aims to make supervised machine learning in Python easy.

Like the R caret package, it provides a consistent API to quickly an easily train and evaluate models. It currently provides wrapper classes around over 20 classification and regression models from the sklearn and xgboost packages.

The bulk of the work is done in the Train object. The TrainControl object specifies the resampling scheme, and is passed to an object of class Train which specifices the learning algorithm, candidate hyperparameters, training samples and the evaluation metric. Keword arguments can be passed directly to the underlying alogorithm.

Once trained, the hyperparameter choices can be easily visualised against the performance metric using Train.plot(). A predict method is available to predict new samples.

The underlying algorithm is always stored as an attribute to the Train object, and is therefore easy to access.

pycaret emphasises usage of the Pandas dataframe. This is because working with dataframes is intuitive and tends to be more accessable to users than numpy arrays.

### Usage information

pycaret has been tested using python3.5. It was tested with the following packages:

- sklearn 0.18.1
- numpy 1.12.1
- pandas 0.19.2
- matplotlib 2.0.0
- seaborn 0.7.1
- scipy 0.19.0

### Installation instructions

For the current version (0.0.1.dev1) it is recommended to create an anaconda environment:

``` bash
# use anaconda3
conda create --name pycaret-env python=3.5

# activate conda environment
source activate pycaret-env

#install dependencies
conda install numpy
conda install scipy
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install seaborn

# clone the repository
git clone https://github.com/philipmgoddard/pycaret.git

# checkout and install
# I found python setup.py install to be buggy when using anaconda
cd pycaret
git checkout develop
pip install ../pycaret/
```

### Feedback

pycaret is under active development. Please feel free to submit improvements, issues, and contribute! Once stable, it will be released on PyPi.





