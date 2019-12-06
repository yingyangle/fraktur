# Christine Yang
# Fraktur Cracker
# majority.py
# get majority baseline to compare with our model

import os, operator
from collections import Counter
from os.path import join


your_path_here = '/Users/Christine/cs/fraktur/'
# your_path_here = '/Users/ovoowo/Desktop/fraktur/'

# get majority baseline accuracy
def getMajorityBaseline(folderpath):
    images = [x for x in os.listdir(folderpath) if x[-3:] == 'png']
    actualLabels = [x[x.rfind('_')+1:-4] for x in images]
    countDict = dict(Counter(actualLabels)) # dict of each letter occurrence
    correctCount = 0
    majorityLabel = max(countDict.items(), key=operator.itemgetter(1))[0]
    return countDict[majorityLabel] / len(images)

# get majority baseline accuracy for only 2 letters (binary classification)
def twoLettersMajorityBaseline(folderpath, a, b):
    images = [x for x in os.listdir(folderpath) if x[-3:] == 'png']
    images = [x for x in images if x[-5:-4] == a or x[-5:-4] == b]
    actualLabels = [x[x.rfind('_')+1:-4] for x in images]
    countDict = dict(Counter(actualLabels)) # dict of each letter occurrence
    correctCount = 0
    majorityLabel = max(countDict.items(), key=operator.itemgetter(1))[0]
    return countDict[majorityLabel] / len(images)
    

# execute

# os.chdir(join(your_path_here, 'testdata')) # testing
os.chdir(join(your_path_here, 'data/3books'))

# accuracy = getMajorityBaseline(os.getcwd())
# accuracy = round(accuracy, 3)
# print('Majority Baseline Accuracy:', accuracy)

accuracy = twoLettersMajorityBaseline(os.getcwd(), 'f', 'ſ')
accuracy = round(accuracy, 3)
print("Majority Baseline Accuracy for 'f' and 'ſ':", accuracy)

# results for data/3books
# Majority Baseline Accuracy: 0.17995713730130491
# Majority Baseline Accuracy for 't' and 'k': 0.8349104427171342
# Majority Baseline Accuracy for 'f' and 'ſ': 0.7195010089891763
