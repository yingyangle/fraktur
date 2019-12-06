# Christine Yang
# Fraktur Cracker
# tess.py
# test Tesseract's accuracy against our model

# USING TESSERACT
# https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d
# https://www.quora.com/Is-it-possible-to-output-the-character-word-line-segmentation-using-Tesseract-OCR

# DOWNLOADING TESSERACT
# https://github.com/sirfz/tesserocr/issues/177
# https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
# ^^^ download fraktur file from here and copy to tessdata folder

import os, codecs, numpy as np, tesserocr, cv2
from tesserocr import PyTessBaseAPI
from collections import Counter
from os.path import join


your_path_here = '/Users/Christine/cs/fraktur/'
# your_path_here = '/Users/ovoowo/Desktop/fraktur/'

# print Tesseract's transcription of a Fraktur image
def tess(filename):
    # frk = german fraktur, psm=7 treats image as line of text
    with PyTessBaseAPI(lang='frk', psm=7) as api:
        api.SetImageFile(filename)
        actualLabel = filename[filename.rfind('_')+1:-4]
        predictLabel = api.GetUTF8Text().rstrip()
        print(filename)
        print('Actual:', actualLabel, 'Prediction:', predictLabel, \
               'Confidence:', str(api.AllWordConfidences())) # 0 worst, 100 best
        if actualLabel == predictLabel: 
            print('CORRECT ~~~\n')
            return 1
        else: 
            print('INCORRECT >:(\n')
            return 0

# run tesseract's predictions on file of letter images, get accuracy
def runTessTest(folderpath):
    images = [x for x in os.listdir(folderpath) if x[-3:] == 'png']
    images.sort()
    actualLabels = [x[x.rfind('_')+1:-4] for x in images]
    letterCorrectCounts = [0 for x in actualLabels]
    countDict = dict(Counter(actualLabels))
    correctDict = \
        {actualLabels[i]:letterCorrectCounts[i] for i in range(len(actualLabels))} 
    api = PyTessBaseAPI(lang='frk', psm=7)
    correctCount = 0
    for i in range(len(images)):
        img = images[i]
        api.SetImageFile(img)
        predictLabel = api.GetUTF8Text().rstrip()
        if actualLabels[i] == predictLabel: 
            correctCount += 1
            correctDict[predictLabel] += 1
    accuracy = correctCount / len(images)
    accuracyDict = \
        {x:correctDict[x]/countDict[x] for x in countDict}
    return (accuracy, accuracyDict)
            
        
    
# execute

os.chdir(join(your_path_here, 'data/3books'))
accuracy, accuracyDict = runTessTest(os.getcwd())
print('Tesseract Accuracy:', accuracy)
print('Tesseract Accuracy for each letter:')
for key,value in accuracyDict.items():
    print(key,value,'%')

# # testing
# os.chdir(join(your_path_here, 'testdata'))
# images = [x for x in os.listdir() if x[-3:] == 'png']
# # get Tesseract transcriptions
# for img in images:
#     ans = tess(img)
#     correctCount += ans
#
# print(correctCount, '/', len(images), 'correct')
# print('Accuracy:', correctCount/len(images))
