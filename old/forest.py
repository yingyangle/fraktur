# Christine Yang
# Fraktur Cracker
# forest.py
# train and run Random Forests model on data

import os, pickle, numpy as np, matplotlib.pyplot as plt, sklearn.ensemble
from sklearn.model_selection import train_test_split

your_path_here = '/Users/Christine/cs/fraktur/'
# your_path_here = '/Users/ovoowo/Desktop/fraktur/'

def getData(filename):
    dataset = np.loadtxt(filename, delimiter = ',')
    (numSamples, numFeatures) = dataset.shape
    data = dataset[:,range(numFeatures-1)].reshape((numSamples,  numFeatures-1))
    labels = dataset[:, -1].reshape((numSamples,))
    return (data,labels)

os.chdir(your_path_here)
data, labels = getData('100d8_blacktestdata.txt')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 400, max_features = 3, oob_score = True)
rf.fit(X_train, y_train)
# rf_score = rf.score(X_test, y_test)
rf_predict = rf.predict(X_train)
rf_score = classification_report(y_train, rf_predict, labels=[1,0])
