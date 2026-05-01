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
task. I applied log transformation and StandardScaler to preprocess the data, and 
compared the performance of three models: Logistic Regression, Random Forest, and 
XGBoost. Class imbalance (77% no defect, 23% defect) was handled through class 
weighting parameters in each model.

Our best model was Logistic Regression, achieving an AUC of 0.7827 on the held out 
test set. At the time of writing, the best performance on Kaggle for this competition 
is an AUC of approximately 0.79.
## Summary of Workdone

### Data

* Data:
  * Type: CSV file of 21 numerical software code metrics, output: defects column (True/False, converted to 0/1)
  * Size: train.csv is approximately 8MB, test.csv is approximately 5MB
  * Instances:
    * Train (60%): 61,057 data points
    * Validation (20%): 20,353 data points
    * Test (20%): 20,353 data points
    * Kaggle Test: 67,856 data points (no labels, used for submission)
    #### Preprocessing / Clean up

* All 21 features are numerical, no categorical features, so no one-hot encoding was needed.
* No missing values were found in the dataset.
* Log transformation (`log1p`) was applied to all features to handle heavy right skew 
  and compress extreme outlier values. `log1p` was used instead of `log` because 
  some features contain zero values.
* StandardScaler was applied after log transformation to put all features on the 
  same scale (mean=0, std=1), which is important for Logistic Regression.
* Class imbalance (77% no defect, 23% defect) was handled through model parameters 
  rather than resampling the data:
  * Logistic Regression and Random Forest: `class_weight="balanced"`
  * XGBoost: `scale_pos_weight=3.41` (ratio of no defect to defect samples)
