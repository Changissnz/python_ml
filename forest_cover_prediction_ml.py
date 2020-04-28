"""
Compares algorithmic pipelines used to train forest-cover-type-prediction dataset

ABOUT DATA
Data attributed to Kaggle. Note that the test*.csv files do not contain
output values. This data was originally used for competition, and those
files constitute the test criteria.

Method below uses only the training file for both training and test samples. 

ABOUT ALGORITHM
Preliminary attempt to construct and train a classifier used to predict the data found
in `forest-cover-type-prediction`. Classifier used is scikit-learn's
KNeighborsClassifier.

No regularization used.
----------------------

The procedure is the following:
- scale the input data
- perform feature selection
    - the feature selector that worked best was scikit-learn's SelectPercentile.
    - initially used PCA that produced subpar results.

- selected the classifier KNeighborsClassifier.
- run the classifier 10 times

The above procedure consistently averages at around 80.5% accuracy rate.

"""

import pandas as pd
import os
import time
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2, f_classif

trainingData = "forest-cover-type-prediction/train.csv"

# open up training data
df = pd.read_csv(trainingData)

OUTPUT_COL = "Cover_Type"
INPUT_COL = [x for x in df.columns if x != "Cover_Type"]

def get_accuracy(predicted, actual):
    assert len(predicted) == len(actual)

    c = 0.0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]: c += 1
    return c / len(predicted)

def train_predict_matrix(df):
    # preprocessing
    df = df[df.columns.difference(["Id"])]
    x = np.asarray(df[df.columns.difference([OUTPUT_COL])])
    y = np.asarray(df[OUTPUT_COL])

    # apply standard scaler

    ss= StandardScaler()
    ss.fit(x)
    x = ss.transform(x)


    # select top 25% of features
    selector = SelectPercentile(f_classif, percentile = 17)
    x = selector.fit_transform(x, y)

    # declare classifier and run training
    def train_predict(x, y, clf):
        x, x2, y, y2 = train_test_split(x, y, test_size = 0.3)
        clf.fit(x, y)

        predicted = clf.predict(x2)
        accuracy = get_accuracy(predicted, y2)
        print("accuracy: ", accuracy)
        return accuracy

    clf = neighbors.KNeighborsClassifier(5, weights="distance")
    c = 0
    for i in range(10):
        c += train_predict(x, y, clf)
    average = c / 10

    print("average accuracy: ", average)
    return average

x = train_predict_matrix(df)
