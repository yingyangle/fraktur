"""
Created on Sun Nov 28 13:25:33 2019
@author: ovoowo
Yuezhen Chen
"""

import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from skimage import exposure
import cv2
import matplotlib.pyplot as plt

your_path_here = '/Users/ovoowo/Desktop/fraktur'
# your_path_here = '/Users/Christine/cs/fraktur'

file =
dataset = np.loadtxt(file,delimiter=',')
(numSamples, numFeatures) = dataset.shape
data = dataset[:,range(2)].reshape((numSamples, 2)) #vector X
# print(data)
labels = dataset[:, 2].reshape((numSamples,))

# take the data and construct the training and testing split, using 75% of the
# data for training and 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(data,
labels , test_size=0.25, random_state=42)

# show the sizes of each data split

print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
          # train the k-Nearest Neighbor classifier with the current value of `k`
          clf = KNeighborsClassifier(n_neighbors=k)
          lfc.fit(trainData, trainLabels)
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
print(Evaluation on Test Data")
print(classification_report(testLabels, predictions))

# loop over a few random digits
for i in np.random.randint(0, high=len(testLabels), size=(5,)):
         # grab the image and classify it
         image = testData[i]
         prediction = model.predict([image])[0]
         # show the prediction
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print("Model predict the letter is : {}".format(prediction))
         #cv2.imshow("image", image)
         plt.show()
         cv2.waitKey(0)
