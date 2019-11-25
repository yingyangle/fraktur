#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:51:33 2019

@author: ovoowo
"""

import cv2
from skimage.feature import hog
import numpy as np
import os

os.chdir('/Users/ovoowo/Desktop/fraktur/segmentation/letters/E')


def generate_txt_image():
    #convert the image to 1s and 0s
    img = cv2.imread('hoff_25_e.png',cv2.IMREAD_GRAYSCALE)
    matimg=np.array(img)
    print(matimg)
    bn = np.where(matimg >= 75, 1, 0)
    (row, col) = bn.shape
#    print(bn.shape)
    for i in range(row):        
        print(bn[i])
    print(bn.shape)
    
    t = [np.hsplit(bn,4)[i] for i in list(range(0, 4))]
    print(t[1].shape)
 #   print(' \n', t)
    block = [t[i][:-(row%4)] for i in list(range(0,len(t)))]
    print(np.array(block).shape)
    r = [t[i][-(row%4):] for i in list(range(0,len(t)))]
  #  print(r)
    print(np.array(r).shape)
    newt = [np.vsplit(block[i],4) for i in list(range(0, 4))]
    
    print(len(newt),len(newt[0]), newt[0][0].shape)
    #print('Segmented: \n  ', newt) #+str(newt[0].shape)
 #   print(newt[0][3])
    
    #concatanate
    #contains section 3, 7, 11, 15
    y = [np.concatenate((newt[i][3], r[i]), axis=0) for i in list(range(0, 4))] 
  #  print('y = \n', y)
    temp = []
    for i in range(4):
        for j in range(4):
            if j != 3:
                temp.append(newt[i][j])
            else:
                temp.append(y[i])
    print('sections = \n', temp)
#    secs = []
#    for i in range(4):
#        sec = [np.hsplit(t[i],4)[i] for i in list(range(0, 4))]
#        print(sec)

 #   for i in range
#    for i in range(t1[1].shape[1]):        
#        print(t1[i])
#    print(t1.shape())
generate_txt_image()

def sectiondivisior(bn):
    (row, col) = bn.shape
    if col % 4 == 0:
        t = [np.hsplit(bn,4)[i] for i in list(range(0, 4))]
        if row % 4 == 0:
            sec = [np.hsplit(t[i],4)[i] for i in list(range(0, 4))]
        else:
            block = [t[i][:-(row%4)] for i in list(range(0,len(t)))]
            r = [t[i][-(row%4):] for i in list(range(0,len(t)))] #storing the residules of row%4
            newt = [np.vsplit(block[i],4) for i in list(range(0, 4))]
             #concatanate
             #contains section 3, 7, 11, 15
            y = [np.concatenate((newt[i][3], r[i]), axis=0) for i in list(range(0, 4))]
            temp = []
            for i in range(4):
                for j in range(4):
                    if j != 3:
                        temp.append(newt[i][j])
                    else:
                        temp.append(y[i])
