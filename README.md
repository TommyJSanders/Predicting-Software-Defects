# Software Defect Detection: A Binary Classification

This repository holds an attempt to apply Logistic Regression, Random Forest, 
and XGBoost to predict software defects using data from the 
[Binary Classification with a Software Defects Dataset](https://www.kaggle.com/competitions/playground-series-s3e23) 
Kaggle challenge.
## Overview

The task, as defined by the Kaggle challenge, is to use 21 software code metrics 
to predict whether a C program module contains a defect or not. The features are 
derived from McCabe and Halstead complexity measures, which capture properties like 
cyclomatic complexity, lines of code, and estimated programming effort.

The approach in this repository formulates the problem as a binary classification 
task. We applied log transformation and StandardScaler to preprocess the data, and 
compared the performance of three models: Logistic Regression, Random Forest, and 
XGBoost. Class imbalance (77% no defect, 23% defect) was handled through class 
weighting parameters in each model.

Our best model was Logistic Regression, achieving an AUC of 0.7827 on the held out 
test set. At the time of writing, the best performance on Kaggle for this competition 
is an AUC of approximately 0.79.
