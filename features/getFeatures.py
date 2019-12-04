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
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'

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
    temp = filename[:-4]
    id = 0
    while id != -len(filename):
        id -= 1
        if filename[i] =='_':
            temp = filename[i+1:]
            break
    label = temp
    ############unfinished
    label =np.array([ord(label)])
    return (black,dist,label)
a ='ch'
len(a)
# execute
#datapath to folder that store image
def txtGenerator(storepath,datapath,mode,foldername,n): #mode = 0 no label, mode = 1 with labels
    os.chdir(datapath)
    Bdataset = []
    Ddataset = []
    letters = []
    nImg = len(os.listdir())
    tracker = 0
    exceptions = [] #store error images

    #Error handler
    name = 'errors.txt'
    aus = open(storepath+foldername+name, 'w')
    aus.close()

    for filename in [x for x in os.listdir() if x[-3:] == 'png']:
        tracker += 1
        try:
            letters.append(filename[-5:-4])
            (black,dist,label) = getFeats(filename,n)
            if mode == 1: # with label
                Bdata = np.concatenate((black,label))
                Ddata = np.concatenate((dist,label))
                Bdataset.append(Bdata)
                Ddataset.append(Ddata)
            else:
                Bdataset.append(black)
                Ddataset.append(dist)
        except:
            exceptions.append(filename)
            aus = open(your_path_here+'/'+name, 'a')
            aus.write(filename + '\n')
            aus.close()
        print(str(tracker)+' images/ '+str(nImg)+' images done')
    freq ={} #Get the frequency
    for l in letters:
        keys = freq.keys()
        if l in keys:
            freq[l] += 1
        else:
            freq[l] = 1
    print('Total number of chars = ',len(freq.keys()))
    print ("Char - frequency : \n", freq)
    Btestdata = np.array(Bdataset)
    Dtestdata = np.array(Ddataset)
    print('Feature extraction for '+str(tracker)+' images done')
    print('='*40+'\n'+'Error images:\n')
    temp = [print(x) for x in exceptions]
    print('Total '+str(len(exceptions))+' Error \n'+'='*40)

    np.savetxt(storepath+foldername+str(n)+'_b.txt',Btestdata, delimiter=', ', fmt='%12.8f')
    np.savetxt(storepath+foldername+str(n)+'_d.txt',Dtestdata, delimiter=', ', fmt='%12.8f')
    return
