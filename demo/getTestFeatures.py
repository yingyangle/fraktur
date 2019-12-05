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

import os,pickle, numpy as np, cv2
from zoning_YC import blackPerSect, blackPerImg, getDistance


# get features for a char image
def getFeats(datapath,filename,n):
    your_path_here = '/Users/ovoowo/Desktop/fraktur/'
    os.chdir(your_path_here+'demo/')
    customdict = pickle.load(open('dictionary.sav','rb'))
    os.chdir(datapath)
    img = cv2.imread(filename)
    num_rows, num_cols, _ = img.shape
    size = np.array([num_rows / num_cols]) # width/height ratio of image
#    blackS = blackPerSect(filename) # list of black ratios for each section
    black = blackPerImg(filename,n) # list of black ratios for each section over the whole image
#    dist = getDistance(filename,n) # edge to char distance
    temp = filename[:-4]
    id = 0
    while id != -len(temp):
        id -= 1
        if temp[id] =='_':
            temp = temp[id+1:]
            break
    ############finished#############
    if len(temp) != 1:
        label = np.array(customdict[temp])
    else:
        label =np.array([ord(temp)])
    return (black,label)

def getData(filename,txtpath):
    os.chdir(txtpath)
    dataset = np.loadtxt(filename, delimiter = ',')
    #dataset = np.loadtxt(Dfilename, delimiter = ',')
#    (numSamples, numFeatures) = dataset.shape
    data = dataset[:-1]
    label=dataset[-1]
    return (data,label)
