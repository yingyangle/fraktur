#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:51:33 2019
@author: ovoowo

Yuezhen Chen & Christine Yang
Fraktur Cracker
zoning.py
section a char image into 16 sections, indexed as follows:

_________________________________
|       |       |       |       |
|   0   |   4   |   8   |   12  |
|       |       |       |       |
_________________________________
|       |       |       |       |
|   1   |   5   |   9   |   13  |
|       |       |       |       |
_________________________________
|       |       |       |       |
|   2   |   6   |   10  |   14  |
|       |       |       |       |
_________________________________
|       |       |       |       |
|   3   |   7   |   11  |   15  |
|       |       |       |       |
_________________________________

"""

import os, numpy as np, cv2

your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/segmentation/letters/E')

    
THRESHOLD = 75 # adjustable threshold for b/w binary image

# split binary image into 16 sections, return list of section matrices
def getSections(binary_img):
    num_rows, num_cols = binary_img.shape
    sects = []
    if num_cols % 4 == 0: # if number of columns divisible by 4
        cols = np.hsplit(binary_img, 4)
    else: # if number of columns NOT divisible by 4
        block = binary_img[:,:num_cols//4*4] # block of 4 evenly divided cols, w/ remainder left out
        cols = np.hsplit(block, 4) # split even block into 4 even col sections
        remainder = np.array([binary_img[:,num_cols//4*4:]]) # leftover cols
        for rem in remainder.T: # for each remainder col, add it to the last col
            cols[-1] = np.concatenate((cols[-1], rem), axis=1)
    if num_rows % 4 == 0: # if number of rows divisible by 4
        sects = [np.vsplit(cols[i], 4) for i in range(4)]
        return [item for sublist in sects for item in sublist]
    else: # if number of rows NOT divisible by 4
        for col in cols: # for each col section
            block = col[:num_rows//4*4] # block of 4 evenly divided rows, w/ remainder left out
            rows = np.vsplit(block, 4) # split even block into 4 even row sections
            remainder = np.array(col[num_rows//4*4:]) # leftover rows
            for rem in remainder: # for each remainder crow, add it to the last row
                rows[-1] = np.concatenate((rows[-1], np.array([rem])), axis=0)
            sects.append(rows)
    return [np.array(sects).flatten()]

def blackPerSect(sects):
    percentages = []
    for i in range(len(sects)):
        num_nonzero = np.count_nonzero(sects[i])
        secsize = sects[i].size
        percentage = (secsize - num_nonzero)/secsize
        percentages.append(percentage)
    print('\n The percentage of blackness per section = \n', percentages)
    return np.array(percentages)


def blackPerImg(sects, binary_img):
    percentages = []
    for i in range(len(sects)):
        num_nonzero = np.count_nonzero(sects[i])
        secsize = sects[i].size
        imgsize = binary_img.size
        percentage = (secsize - num_nonzero)/imgsize
        percentages.append(percentage)
    print('\n The percentage of blackness per image = \n', percentages)
    return np.array(percentages)

# get image, convert to binary, split into 16 sections, print sections
def printSections(filename):
    # convert the image to 1s and 0s
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    matrix = np.array(img) # np matrix of img vals
    binary_img = np.where(matrix >= THRESHOLD, 1, 0) # convert grayscale img to binary
    temp = [print(x) for x in binary_img]
    sects = getSections(binary_img) # list of matrices for each section
    for s in range(len(sects)):
        print('\nsection {}'.format(s), '{0:#^20}'.format(''))
        print(sects[s])
    
    return (sects, binary_img)

# execute

filename = 'hoff_25_e.png'
(sects, binary_img) = printSections(filename)
blackPerSect(sects)
blackPerImg(sects, binary_img)
#print(type(sects[0]))
#print(type(sects[0]))
