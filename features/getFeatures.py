# Christine Yang, Yuezhen Chen
# Fraktur Cracker
# getFeatures.py
# get features for each char image

# https://pdfs.semanticscholar.org/6d50/6c0c85cda0ab43b47f997d8c179986e1ba5a.pdf

# features include:

# - zoning: split image into 16 sections
#       - black pixels in section / # total pixels in section
#       - black pixels in section / # total pixels in whole image
# - height/width ratio of image
# - distance profile features
#       - # pixels (distance) from upper edge of image to outer edge of char
#       - # pixels (distance) from lower edge of image to outer edge of char
#       - # pixels (distance) from left edge of image to outer edge of char
#       - # pixels (distance) from right edge of image to outer edge of char

import os, numpy as np, cv2
from zoning_YC import blackPerSect, blackPerImg, getDistance
import pandas as pd
import csv
# from distance import getDistance

your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/letters_for_testing')

# get features for a char image
def getFeats(filename,n):
    feats = np.array([])
    img = cv2.imread(filename)
    num_rows, num_cols, _ = img.shape
    size = np.array([num_rows / num_cols]) # width/height ratio of image
#    blackS = blackPerSect(filename) # list of black ratios for each section
    black = blackPerImg(filename,n) # list of black ratios for each section over the whole image
    dist = getDistance(filename,n) # edge to char distance
    # feats = np.concatenate((size, black, dist))
    label =np.array([ord(filename[-5:-4])])
    # print('label = ',label)
    # data = np.concatenate((feats,label))
    # print('data = ',data)
    # return data
    return (black,dist,label)


# execute
Bdataset = []
Ddataset = []
letters = []
for filename in [x for x in os.listdir() if x[-3:] == 'png']:
    letters.append(filename[-5:-4])
    (black,dist,label) = getFeats(filename,8)
    Bdata = np.concatenate((black,label))
    Ddata = np.concatenate((dist,label))
    Bdataset.append(Bdata)
    Ddataset.append(Ddata)
freq ={}
for l in letters:
    keys = freq.keys()
    if l in keys:
        freq[l] += 1
    else:
        freq[l] = 1
print(freq)
print(len(freq.keys()))
Btestdata = np.array(Bdataset)
Dtestdata = np.array(Ddataset)
np.savetxt('blacktestdata.txt',Btestdata, delimiter=', ', fmt='%12.8f')
np.savetxt('distancetestdata.txt',Dtestdata, delimiter=', ', fmt='%12.8f')
#dict = {'features': features}
#df = pd.DataFrame(dict)
#df.to_csv('testdata.csv', header=False, index=False, sep=",",escapechar=" ",quoting=csv.QUOTE_NONE)
