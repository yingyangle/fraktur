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
'''=====================================================
Helpter function
======================================================'''
def sectiondivisior(bn):
    (row, col) = bn.shape
    if col % 4 == 0: #if column is divisible by 4
        t = [np.hsplit(bn,4)[i] for i in list(range(0, 4))]
        if row % 4 == 0: #if row is divisible by 4
            sec = [np.hsplit(t[i],4)[i] for i in list(range(0, 4))]
        else: #if row is not divisible by 4
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
    return temp
    #else: #if column is not divisible by 4
     #   block = [t[i][:-(row%4)] for i in list(range(0,len(t)))]
'''=====================================================
Main section: Please note that the the indices of temp is going down columns...
======================================================'''
def generate_txt_image():
    #convert the image to 1s and 0s
    img = cv2.imread('hoff_25_e.png',cv2.IMREAD_GRAYSCALE)
    matimg=np.array(img)
    print(matimg)
    bn = np.where(matimg >= 75, 1, 0)
    temp = sectiondivisior(bn)
    print('sections = \n', temp)
generate_txt_image()


        
                        
