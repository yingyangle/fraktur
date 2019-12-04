"""
Created on Sun Nov 28 13:25:33 2019
@author: ovoowo
Yuezhen Chen
"""
import matplotlib.image as mpimg
from PIL import Image
import os, math, pickle, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from KNN import getData
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
datapath = your_path_here+'fraktur/data/datatxt'

def knnClf(data, labels,kbound):
# take the data and construct the training and testing split, using 75% of the
# data for training and 25% for testing

    (trainData, testData, trainLabels, testLabels) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
    test_size=0.1, random_state=84)

    # show the sizes of each data split

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k

    kVals = range(1, kbound, 2)
    accuracies = []

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, kbound, 2):
              # train the k-Nearest Neighbor classifier with the current value of `k`
              clf = KNeighborsClassifier(n_neighbors=k)
              clf.fit(trainData, trainLabels)
              # evaluate the model and update the accuracies list
              score = clf.score(valData, valLabels)
              print("k=%d, accuracy=%.2f%%" % (k, score * 100))
              accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))

    # re-train our classifier using the best k value and predict the labels of the
    # test data
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits

    print('\nEvaluation on Test Data')
    charLabels = np.array([chr(int(testLabels[i])) for i in range(testLabels.size)])
    charPredictions = np.array([chr(int(predictions[i])) for i in range(predictions.size)])
    print(classification_report(charLabels, charPredictions))

    # loop over a few random digits
    # for ind in np.random.randint(0, high=len(testLabels), size=(10,)):
    #          # grab the image and classify it
    #          image = testData[ind]
    #          prediction = int(model.predict([image])[0])
    #          print("i think tha letter is : {}".format(chr(prediction))+'\nThe actual value is :'+chr(int(testLabels[i])))
    #store the best model as pickle
    (numSamples, numFeatures) = data.shape
    N = int(math.sqrt(numFeatures)) if math.sqrt(numFeatures).is_integer() else int(numFeatures/4)
    name = str(kVals[i])+'_'+str(N)+'_NN.sav'
    pickleModel = open(your_path_here+'fraktur/demo/'+name,'wb')
    pickle.dump(model, pickleModel)
    Aname = str(kVals[i])+'_'+str(N)+'_accuracy.sav'
    pickleAccuracies = open(your_path_here+'fraktur/demo/'+Aname,'wb')
    pickle.dump(accuracies, pickleAccuracies)
    return accuracies
