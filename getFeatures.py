# Christine Yang
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
from zoning import blackPerSect, blackPerImg
from distance import getDistance

filename = 'hoff_21_e.png'

# get features for a char image
def getFeats(filename):
    img = cv2.imread(filename)
    num_rows, num_cols = img.shape
    blackS = blackPerSect(filename) # list of black ratios for each section
    blackI = blackPerImg(filename) # list of black ratios for each section over the whole image
    size = num_rows / num_cols # height/width ratio of image
    dist = getDistance(filename) # edge to char distance
    return [blackS, blackI, size, dist]

# execute
