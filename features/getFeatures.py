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
# from distance import getDistance

your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/letters_for_testing')


# get features for a char image
def getFeats(filename):
    feats = np.array([])
    img = cv2.imread(filename)
    num_rows, num_cols, _ = img.shape
    size = np.array([num_rows / num_cols]) # width/height ratio of image
    blackS = blackPerSect(filename) # list of black ratios for each section
    blackI = blackPerImg(filename) # list of black ratios for each section over the whole image
    dist = getDistance(filename) # edge to char distance
    #feats = np.concatenate((size, blackS, blackI))
    return (size,blackI,dist)


# execute
sizes = []
blackIs = []
dists = []
bugcheck = 0
for filename in [x for x in os.listdir() if x[-3:] == 'png']:
    bugcheck += 1
    print('This is the '+ str(bugcheck)+' image',filename)
    size,blackI,dist = getFeats(filename)
    sizes.append(size)
    blackIs.append(blackI)
    dists.append(dist)

dict = {'size': sizes, 'blackness': blackIs, 'distance': dists}
df = pd.DataFrame(dict)
df.to_csv('testdata.csv', header=False, index=False)
