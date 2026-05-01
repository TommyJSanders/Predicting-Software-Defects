# Software Defect Detection: A Binary Classification Problem

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
#### Data Visualization

* A histogram grid was plotted comparing the distribution of each feature between 
  the two classes (defect vs no defect) before and after log transformation. 
<img width="659" height="574" alt="Screenshot 2026-05-01 at 1 17 35 AM" src="https://github.com/user-attachments/assets/bdbdfb4d-926c-454e-9bcd-35488e14b1d9" />
Before transformation, most features showed a single spike near zero with a long 
  right tail due to extreme outliers.

<img width="660" height="590" alt="Screenshot 2026-05-01 at 1 19 44 AM" src="https://github.com/user-attachments/assets/2a528d8e-f0b8-46d9-aec4-f1124e57669f" />
After log transformation the distributions 
  became much more spread out and readable, clearly showing differences between 
  the two classes for many features.
  
* A bar chart was plotted showing the class distribution, confirming the imbalance 
  of 77% no defect and 23% defect.
<img width="557" height="370" alt="Screenshot 2026-05-01 at 1 21 19 AM" src="https://github.com/user-attachments/assets/2a28441c-f12f-463e-8159-b65169cd9b4b" />

* A correlation matrix was computed to identify relationships between features. 
  Several features were found to be highly correlated, for example `e` and `t` 
  are mathematically related (t = e/18), and `n` is derived from `total_Op` and 
  `total_Opnd`. The features most correlated with the target were `loc` (0.34), 
  `branchCount` (0.32), and `v(g)` (0.30), suggesting that larger and more complex 
  modules are more likely to contain defects.
  <img width="657" height="572" alt="Screenshot 2026-05-01 at 1 22 46 AM" src="https://github.com/user-attachments/assets/a31142bd-506b-40ad-b3a2-e2681b51cf57" />


  
