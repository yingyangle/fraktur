#!/Users/Christine/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Christine Yang
# Fraktur Cracker
# wordSeg.py
# word segmentation

import os, codecs, numpy as np, cv2, re, sys, glob, more_itertools as mit
from PIL import Image
from preprocess import preprocess

# get correct labels from .txt transcription
def getLabels(filename):
    txt_file = filename[:-4]+'.gt.txt' # get .txt filename
    txt_file = re.sub('\.nrm', '', txt_file)
    ein = open(txt_file, 'r') # open .txt file
    raw = ein.read().rstrip() # read .txt file
    ein.close()
    txt = re.sub(u'[.,\'\"“„-]', '', raw) # replace punctuation
    # add # chars for extra letters at the end from bad segmentation
    word_ls = txt.split(' ') + ['#', '#', '#', '#', '#']
    return word_ls # list of letter labels

def wordSeg(filename, destpath):
    labels = getLabels(filename)
    im = Image.open(filename, 'r') # open image
    width, height = im.size # image size
    pix_val = list(im.getdata()) # pixel color values

    cols = [pix_val[n::width] for n in range(width)] # pixels as columns
    # print('\n'.join(' '.join(map(str,sl)) for sl in cols)) # print cols

    # get column indices of whitespace
    whitespace = []
    for i in range(len(cols)):
        c = cols[i]
        whites = [x for x in c if x >= 200] # whitespace pixels
        if len(whites) >= len(c): whitespace.append(i) # allow W non-whitespace
        # all(x >= 200 for x in c)

    # group consecutive whitespace
    groups = [list(group) for group in mit.consecutive_groups(whitespace)]
    groups_len = [len(g) for g in groups]
    num_spaces = len(labels) - 6
    spaces_i = sorted(range(len(groups_len)), key = lambda sub: groups_len[sub])[-num_spaces:] 
    spaces_i.sort()
    spaces = [groups[i] for i in spaces_i]
    
    os.chdir(destpath)
    # first word
    area = (0, 0, spaces[0][0], height)
    cropped_im = im.crop(area)
    imagename = filename[:-4]+'_0_'+labels[0]+'.png'
    cropped_im.save(imagename)
    for i in range(len(spaces)-1): # crop letter images
        left = spaces[i][-1]
        right = spaces[i+1][0]
        area = (left, 0, right, height)
        cropped_im = im.crop(area)
        # cropped_im.show()
        imagename = filename[:-4]+'_'+str(i)+'_'+labels[i+1]+'.png'
        cropped_im.save(imagename)
    return

# os.chdir('/Users/Christine/Documents/cs/fraktur/segmentation/test_data')
filename = 'word.png'
filename = 'word_1_eine.png'
# getWords(filename, os.getcwd())
a = cv2.imread('word_1_eine.png')
# preprocess(filename)