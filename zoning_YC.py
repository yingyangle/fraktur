#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:51:33 2019
@author: ovoowo

Yuezhen Chen & Christine Yang
Fraktur Cracker
zoning.py
section a char image into 16(nxn) sections, indexed as follows:

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
os.chdir(your_path_here+'fraktur/features/')


THRESHOLD = 75 # adjustable threshold for b/w binary image
n = 4 # adjustable threshold for zoning


# get image and change to binary
def getImg(filename,n):
    # convert the image to 1s and 0s
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    matrix = np.array(img) # np matrix of img vals
    binary_img = np.where(matrix >= THRESHOLD, 1, 0) # convert grayscale img to binary
    (row,col) = binary_img.shape
    #### resizedImg
    if col//n ==0:
        r = (n-col)//2 #0
        l = n - col - r #1
        #print('n = '+str(n)+'\ncol = '+str(col)+'\nr = '+str(r)+'\nl = ',l)
        if(r!=0):
            a = binary_img
            b = np.where(np.zeros((row,col+r))==0,1,0)
            b[:,:-r] = a
        else:
            b = binary_img
        #print(b.shape)
        resizedImg = np.where(np.zeros((row,n))==0,1,0)
        #print(resizedImg.shape)
        resizedImg[:,l:] = b
        #print('resizedImg =\n',resizedImg)
        binary_img = resizedImg
    ####
    return binary_img

# split binary image into 16 sections, return list of section matrices
def getSections(binary_img,n):
    num_rows, num_cols = binary_img.shape
    sects = []
    if num_cols % n == 0: # if number of columns divisible by 4
        cols = np.hsplit(binary_img, n)
    else: # if number of columns NOT divisible by 4
        block = binary_img[:,:num_cols//n*n] # block of 4 evenly divided cols, w/ remainder left out
        cols = np.hsplit(block, n) # split even block into 4 even col sections
        remainder = np.array([binary_img[:,num_cols//n*n:]]) # leftover cols
        for rem in remainder.T: # for each remainder col, add it to the last col
            cols[-1] = np.concatenate((cols[-1], rem), axis=1)
    if num_rows % n == 0: # if number of rows divisible by 4
        sects = [np.vsplit(cols[i], n) for i in range(n)]
        return [item for sublist in sects for item in sublist]
    else: # if number of rows NOT divisible by 4
        for col in cols: # for each col section
            block = col[:num_rows//n*n] # block of 4 evenly divided rows, w/ remainder left out
            rows = np.vsplit(block, n) # split even block into 4 even row sections
            remainder = np.array(col[num_rows//n*n:]) # leftover rows
            for rem in remainder: # for each remainder crow, add it to the last row
                rows[-1] = np.concatenate((rows[-1], np.array([rem])), axis=0)
            sects.append(rows)
    return np.array(sects).flatten()

# split image into 16 sections and print sections
def printSections(filename,n):
    binary_img = getImg(filename,n)
    temp = [print(x) for x in binary_img] # print full image
    sects = getSections(binary_img) # list of matrices for each section
    for s in range(len(sects)): # print each image section
        print('\nsection {}'.format(s), '{0:#^20}'.format(''))
        print(sects[s])
    return (sects, binary_img)

# get black percentage over each section for list of sections
def blackPerSect(filename,n):
    binary_img = getImg(filename,n)
    sects = getSections(binary_img,n)
    percentages = [] # list of final black percentages
    for i in range(len(sects)): # calculate pct for each sect
        sect_size = sects[i].size
        percentage = (sect_size - sects[i].sum()) / sect_size
        percentages.append(percentage)
    #print('\nPercentage Blackness Over Each Section:')
#    temp = [print(x) for x in percentages]
    return np.array(percentages)

# get black percentage over whole image for list of sections
def blackPerImg(filename,n):
    binary_img = getImg(filename,n)
    sects = getSections(binary_img,n)
    percentages = [] # list of final black percentages
    for i in range(len(sects)): # calculate pct for each sect
        img_size = binary_img.size
        percentage = (sects[i].size - sects[i].sum()) / img_size
        percentages.append(percentage)
    # print('\nPercentage Blackness Over Whole Image:')
    # temp = [print(x) for x in percentages]
    return np.array(percentages)

'''
================================================================================
Edge to contour Distance Feature
================================================================================
Top Indices: [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11], [12, 13, 14, 15]]
Bottom Indices: [[ 3,  2,  1,  0],[ 7,  6,  5,  4], [11, 10,  9,  8], [15, 14, 13, 12]]
Left Indices: [[ 0,  4,  8, 12], [ 1,  5,  9, 13], [ 2,  6, 10, 14], [ 3,  7, 11, 15]])
Right Indices: [[12,  8,  4,  0], [13,  9,  5,  1], [14, 10,  6,  2], [15, 11,  7,  3]])
_________________
| 0 | 4 | 8 | 12 |
_________________
| 1 | 5 | 9 | 13 |
_________________
| 2 | 6 | 10| 14 |
_________________
| 3 | 7 | 11| 15 |
_________________

'''
    #Top Edge Distance: First 0 in[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]
    #Bottom Edge Distance: Last 0 in[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]
    #Parameter for getDirectionDist: subsects = [vertical_sects, horizontal_sects]
    #ind = [0,1]: 0 for top/bot, 1 for right/left
    #tbcode, 1 for top distance, -1 for bot distance
def getDirectionDist(subsects, ind, tbcode,n):
    dists = []
    for i in range(n):
        movesect = 0 if tbcode == 1 else 3
        if ind == 0:
            # #print('i ='+str(i)+' movesect =', movesect)
            (row, col) = subsects[i][movesect].shape #Shape for each section
            temp = subsects[i][movesect][:,int(col/2)] #first [] is for which verticle strips of sections(4)
                                    #second [] for which subsection in each verticle strips
                                    #the [:,int(col/2)] can be changed to adjust which distance you want
        else:
            (row, col) = subsects[i][movesect].shape
            temp = subsects[i][movesect][int(row/2)]
        firstZeroInd = np.where(temp == 0)[0]
        dist = 0
        while (firstZeroInd.size == 0):
            shape = subsects[i][movesect].shape
            movesect += tbcode
            if movesect in [-1, n]:
                break
            dist += shape[ind] #top bottom shape = row = 0, right,left shape = col =1
            if ind == 0:
                temp = subsects[i][movesect][:,int(col/2)]
            else:
                temp = subsects[i][movesect][int(row/2)]
            firstZeroInd = np.where(temp == 0)[0]
        if firstZeroInd.size == 0: #in case there's no blackness in that row/column
            dists.append(dist)
        else:
            if tbcode == 1:
                dists.append(dist+firstZeroInd[0])#store the first zero we find
            else:
                shape = subsects[i][movesect].shape
                dists.append(dist+(shape[ind] - firstZeroInd[-1]-1)) #row for bottom, col for right
    return np.array(dists)


def getDistance(filename,n):
    binary_img = getImg(filename,n)
    (row,col) = binary_img.shape
    sects = getSections(binary_img,n)
    top_indices = np.arange(n*n).reshape((n, n))
    top_indices[0]
    left_indices = np.transpose(top_indices)

    vertical_sects = []
    for i in range(n):
        vertical = []
        for j in range(n):
            vertical.append(sects[top_indices[i][j]])
        # vertical_sects.append(np.array(vertical))
        vertical_sects.append(vertical)


    h_sects  = []
    for i in range(n):
        horizontal = []
        for j in range(n):
            a = left_indices[i][j]
            horizontal.append(sects[a])
        h_sects.append(horizontal)

    topdistsPixel = getDirectionDist(vertical_sects, 0, 1,n)
    botdistsPixel = getDirectionDist(vertical_sects, 0, -1,n)
    leftdistsPixel = getDirectionDist(h_sects, 1, 1,n)
    rightdistsPixel = getDirectionDist(h_sects, 1, -1,n)

    # #print(' \nFor our reference:\ntop distance in pixels = ' + str(topdistsPixel)
    # + '\nbot distance in pixels = ' +str(botdistsPixel) + '\nleft distance in pixels = '
    #  +str(leftdistsPixel)+ '\nright distance in pixels = '
    #   +str(rightdistsPixel))

    distances = np.concatenate((topdistsPixel/row, botdistsPixel/row), axis=None)
    distances = np.concatenate((distances,leftdistsPixel/col),axis=None)
    distances = np.concatenate((distances,rightdistsPixel/col),axis=None)
    # #print
    # #print(' \nFor Training:\nScaled top distance = ' + str(distances[0*n:n]) +
    # '\nbot distance = ' +str(distances[n:2*n])+'\nleft distance = ' +str(distances[2*n:3*n])
    # +'\nright distance = ' +str(distances[3*n:4*n]))
    # #print('\nDistances[top, bottom, left, right] =\n',distances)
    return np.array(distances)

# execute
#filename = 'hard2_22_n.png'
#filename = 'hard_5_e.png'
#filename = 'hard_71_e.png'
filename = 'a_4_i.png'

# filename = 'hoff_12_e.png'
binary_img = getImg(filename,n)
sects=getSections(binary_img,n)
getDistance(filename,n)
blackPerSect(filename,n)
blackPerImg(filename,n)
