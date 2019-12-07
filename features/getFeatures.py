# Christine Yang, Yuezhen Chen
# Fraktur Cracker
# getFeatures.py
# get features for each letter image, save features in .txt file

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

import os, numpy as np, pickle
from os.path import join
from shutil import copyfile
from zoning_YC import blackPerSect, blackPerImg, getDistance

# your_path_here = '/Users/ovoowo/Desktop/fraktur/'
your_path_here = '/Users/Christine/cs/fraktur/'


# get features for a char image
# mode=0 get blackness, mode=1 get distance 
def getFeats(datapath, filename, n, mode=2):
    os.chdir(your_path_here+'features/')
    customdict = pickle.load(open('dictionary.sav', 'rb'))
    os.chdir(datapath)
    label = filename[filename.rfind('_')+1:-4] # correct label for letter img
    if len(label) != 1:
        label = customdict[label]
    else:
        label = np.array([ord(label)])
    if mode is 0: # list of black ratios for each section over the whole image
        black = blackPerImg(filename, n) 
        return (black, label)
    if mode is 1: # edge to char distance
        dist = getDistance(filename, n) 
        return (dist, label)


# creat .txt file of distance features for each letter img in a folder
# datapath = folder that stores letter images
# labelMode=0 don't save labels in dataset, labelMode=1 save dataset with labels
# n = zoning size parameter
# featMode=0 get blackness, featMode=1 get distance
def createDataset(storepath, datapath, n, labelMode=1, featMode=0): 
    os.chdir(datapath)
    features = [] # list of features for each letter img
    num_imgs = len(os.listdir()) # number of letter images
    tracker = 0
    for filename in [x for x in os.listdir() if x[-3:] == 'png']:
        tracker += 1
        try: # get feature for this img
            feature = getFeats(datapath, filename, n, featMode)
            if labelMode == 1: # attach label
                features.append(np.concatenate(feature))
            else: # don't attach label
                features.append(feat)
        except Exception as e: # write error to .txt file
            print(e) # print error
            aus = open(storepath+'errors.txt', 'a')
            aus.write(filename + '\n')
            aus.close()
        if num_imgs < 500: # print progress
            if tracker % 100 == 99: 
                print(str(tracker)+' images / '+str(num_imgs)+' images done')
        else: # print progress
            if tracker % 500 == 499: 
                print(str(tracker)+' images / '+str(num_imgs)+' images done')
    dataset = np.array(features)
    # print('Feature extraction for '+str(tracker)+' images done')
    # print('='*40+'\n'+'Error images:\n')
    # print('Total '+str(len(exceptions))+' Error \n'+'='*40)
    foldername = datapath[datapath.rfind('/')+1:]
    np.savetxt(join(storepath,foldername+'_'+str(n)+'zones.txt'),\
                dataset,delimiter=', ',fmt='%12.8f')
    return

# only get imgs from datapeth if their label is in letters list
# copy imgs to new folder
# doesn't work for digraphs at the moment
def getLetters(datapath, letters):
    os.chdir(datapath)
    new_folder = ''.join(letters)
    try:
        os.mkdir(new_folder) # make new folder to store imgs for these letters
    except: pass # folder exists for some reason or another
    images = [x for x in os.listdir() if x[x.rfind('_')+1:-4] in letters]
    for img in images: # copy matching imgs to new folder
        copyfile(img, join(your_path_here, 'data', new_folder, img))
    return

# execute
    
# get matching letter imgs
# datapath = join(your_path_here, 'data/3books')
# getLetters(datapath, ['f', 'ſ'])

# get features dataset
# datapath = join(your_path_here, 'data/fſ')
# storepath = join(your_path_here, 'data')
# createDataset(storepath, datapath, 8, 1, 0)