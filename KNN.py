# Yuezhen Chen, Christine Yang
# Fraktur Cracker
# KNN.py
# train KNN models for multiple values of k, pick best one, return accuracy

import os, pickle, math, numpy as np, matplotlib.image as mpimg
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from os.path import join

your_path_here = '/Users/Christine/cs/fraktur/'
# your_path_here = '/Users/ovoowo/Desktop/fraktur/'


# get feature data from .txt file
def getData(filename, txtpath):
    os.chdir(txtpath)
    dataset = np.loadtxt(filename, delimiter = ',')
    numSamples, numFeatures = dataset.shape
    data = dataset[:,range(numFeatures-1)].reshape((numSamples,  numFeatures-1)) #vector X
    labels = dataset[:, -1].reshape((numSamples,))
    return (data, labels)

# train KNNs for multiple k values, pick best one using validation data
# get accuracies for best model
# kbound = max k value to try
def knnClf(data, labels, kbound):
    # split 75% training, 25% testing data
    trainData, testData, trainLabels, testLabels = train_test_split(data,
    labels, test_size=0.25, random_state=42)
    trainData, valData, trainLabels, valLabels = train_test_split(trainData, trainLabels,
    test_size=0.1, random_state=84) # validation data split
    numSamples, numFeatures = data.shape
    N = int(math.sqrt(numFeatures)) if math.sqrt(numFeatures).is_integer() else int(numFeatures/4)
    
    # show the sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    kVals = range(1, kbound, 2) # values of k to test
    accuracies = [] # accuracies for each k
    Pins = []
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, kbound, 2):
              # train the k-Nearest Neighbor classifier with the current value of `k`
              clf = KNeighborsClassifier(n_neighbors=k)
              clf.fit(trainData, trainLabels)
              Pins.append(clf.score(trainData, trainLabels))
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

    ####Confusion matrix
    charLabels = [chr(int(testLabels[i])) for i in range(testLabels.size)]
    # disp = metrics.plot_confusion_matrix(model, testData, testLabels)
    # show_confusion(model,testLabels, predictions,charLabels,N)

    # print final classification report 
    print('\nEvaluation on Test Data')
    charPredictions = np.array([chr(int(predictions[i])) for i in range(predictions.size)])
    print(classification_report(np.array(charLabels), charPredictions))

    # loop over a few random digits
    # for ind in np.random.randint(0, high=len(testLabels), size=(10,)):
    #          # grab the image and classify it
    #          image = testData[ind]
    #          prediction = int(model.predict([image])[0])
    #          print("i think tha letter is : {}".format(chr(prediction))+'\nThe actual value is :'+chr(int(testLabels[i])))
    
    #store the best model as pickle
    mod_name = 'k'+str(kVals[i])+'z'+str(N)+'_KNN.sav'
    pickleModel = open(join(your_path_here, mod_name),'wb')
    pickle.dump(model, pickleModel)
    acc_name = 'k'+str(kVals[i])+'z'+str(N)+'_accuracy.sav'
    pickleAccuracies = open(join(your_path_here, acc_name),'wb')
    pickle.dump(accuracies, pickleAccuracies)
    return (accuracies, Pins) # best k and its accuracy
    

# execute

# filename = 'fſ_8zones.txt'
# txtpath = join(your_path_here, 'data')
# data, labels = getData(filename, txtpath)
# accuracies, Pins = knnClf(data, labels, 15)