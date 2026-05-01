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

### Problem Formulation

* Input: 21 numerical software code metrics (McCabe and Halstead complexity measures)
* Output: Binary prediction of whether a module contains a defect (0 = no defect, 1 = defect)

* Models:
  * **Logistic Regression** — chosen as a simple baseline model. Benefits  
    from log transformation and StandardScaler. Handles class imbalance via 
    `class_weight="balanced"`.
  * **Random Forest** — chosen because it handles outliers and skewed data  
    and generally performs well on tabular data. Handles class imbalance via 
    `class_weight="balanced"`.
  * **XGBoost** — chosen because boosted decision trees are considered the preferred 
    algorithm for tabular data and a lot of other Kaggle submissions utilized it. Handles class imbalance via `scale_pos_weight`.

* Hyperparameters:
  * Logistic Regression: `max_iter=1000`, `class_weight="balanced"`
  * Random Forest: `n_estimators=100`, `class_weight="balanced"`
  * XGBoost: `n_estimators=100`, `scale_pos_weight=3.41`, `eval_metric="logloss"`
### Training

* All models were trained on a personal MacBook using Python and Jupyter Notebook.
* The following packages were used: pandas, numpy, matplotlib, scikit-learn, and xgboost.
* Training times were fast; Logistic Regression and XGBoost trained in seconds, 
  Random Forest took slightly longer due to building 100 decision trees.
* These models do not have traditional training curves (loss vs epoch) since they 
  are not neural networks
* Training was stopped automatically when the models converged. For Logistic 
  Regression, `max_iter=1000` was set to ensure convergence.
* The main difficulty encountered was installing XGBoost on MacOS, which required 
  installing the OpenMP runtime library via `brew install libomp` before the 
  package could be imported successfully. Even that was pretty straight forward.
### Performance Comparison

* The key performance metrics used are:
  * **ROC AUC** — the primary Kaggle competition metric. Measures the model's 
    ability to distinguish between defect and no defect across all thresholds. 
    A score of 1.0 is perfect, 0.5 is random guessing.
  * **Recall (defect class)** — measures how many actual defects the model caught. 
    This is important because missing a real bug is worse than a false alarm.
  * **F1 Score (defect class)** — balances precision and recall for the defect class.

* Results sliced test set:

<img width="446" height="504" alt="Screenshot 2026-05-01 at 1 32 19 AM" src="https://github.com/user-attachments/assets/74a550cc-d511-4b7a-b212-53a27c85805d" />


* ROC curves were plotted for all three models showing Logistic Regression 
  achieving the highest AUC of 0.7827, closely followed by XGBoost at 0.7703.
<img width="658" height="519" alt="Screenshot 2026-05-01 at 1 32 46 AM" src="https://github.com/user-attachments/assets/36b11d42-9467-44f5-a902-711d336099e8" />
### Conclusions

* Logistic Regression was the best model with an AUC of 0.7827 and defect 
  recall of 0.68, making it the most suitable for this task.
* Random Forest had the highest accuracy (0.81) but worst defect recall (0.35), 
  showing that accuracy is a misleading metric when classes are imbalanced.
* Log transformation significantly improved model performance by reducing skew 
  and compressing extreme outlier values.
* Several features are mathematically derived from others, meaning the dataset 
  contains redundant information that could be removed.


  

  
