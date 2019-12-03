"""
Created on Sun Nov 28 13:25:33 2019
@author: ovoowo
Yuezhen Chen
"""
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt
import os, math
#from getTestFeatures import getFeats
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
datapath = your_path_here+'features/'
os.chdir(datapath)
from getFeatures import getFeats,txtGenerator
'''
Plotting Function
'''
def kAccuracy(ks,accuracies,feature):
    fig = plt.figure()
    plt.plot(np.array(ks), np.array(accuracies), 'c.-')
    fig.show()
    plt.xlabel('K')
    plt.ylabel('Average Error Rate')
    s ='Average Eout vs. K'
    plt.title(s)
    plt.savefig('knnAccuracy_{}.png'.format(feature),dpi=200)
    return

'''
Generate Feature for KNN
'''
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
datapath = your_path_here+'fraktur/data/dataset'
# os.chdir(datapath)
Folder = 'dataset'
txtGenerator(datapath,1,Folder)
'''
Access Features txt file
'''
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/')

Bfilename = Folder+'_blacktestdata.txt'
#Dfilename = Folder+'_distancetestdata.txt'

dataset = np.loadtxt(Bfilename, delimiter = ',')
#dataset = np.loadtxt(Dfilename, delimiter = ',')
(numSamples, numFeatures) = dataset.shape
data = dataset[:,range(numFeatures-1)].reshape((numSamples,  numFeatures-1)) #vector X
labels = dataset[:, -1].reshape((numSamples,))
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

kVals = range(1, 60, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 60, 2):
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
kAccuracy(kVals,accuracies,'black') #or 'dist'

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
for i in np.random.randint(0, high=len(testLabels), size=(10,)):
         # grab the image and classify it
         image = testData[i]
         prediction = int(model.predict([image])[0])
         # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
         # then resize it to 32 x 32 pixels so we can see it better
##         image = image.reshape((64, 64))
##         image = exposure.rescale_intensity(image, out_range=(0, 255))
##         image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

         # show the prediction

         # imgdata = np.array(image, dtype='float')
         # d = int(math.sqrt(numFeatures-1))
         # pixels = imgdata.reshape((d,d))
         # plt.imshow(pixels,cmap='gray')
         # plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print("i think tha letter is : {}".format(chr(prediction))+'\nThe actual value is :'+chr(int(testLabels[i])))

         #cv2.imshow("image", image)
         plt.show()
###for presentation
pickle.dump(model, open('5NN.sav', 'wb'))

def randomPrediction():
    your_path_here = '/Users/ovoowo/Desktop/fraktur/'
    #your_path_here = '/Users/Christine/cs/fraktur/'
    datapath = your_path_here+'data/dataset/'
    files = os.listdir(datapath)
    n = len(files)
    testID = np.random.randint(n-1,size = 10)
    testID[0]
    presentFile = [files[id] for id in testID]
    for file in presentFile:
        img = mpimg.imread(datapath+file) #Importing image data into Numpy arrays
        imgplot = plt.imshow(img)
        (black,dist,label) = getFeats(file,8)
        pred = chr(int(model.predict(black.reshape(1, -1))))
        # pred = model.predict(dist)
        charLabel = chr(int(label))
        plt.title('Label: '+pred+' Prediction: '+pred)
        plt.show()
