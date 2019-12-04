# -*- coding: utf-8 -*-
# Christine Yang
# Fraktur Cracker
# labelData.py
# separates labeled data into folders for each letter

import os, re
from os.path import join


def labelData(datapath, destpath):
    os.chdir(datapath) # path to folder of segmented letter imgs
    # each letter .png in folder (don't include filename_morph, filename_sanity, filename)
    images = [x for x in os.listdir() if x[-3:] == 'png' and x.count('_') > 3]
    # for each letter image in the folder
    for img in images:
        index = img.rfind('_') # index of last occurrence of '_' in filename
        label = img[index+1:-4] # letter label of this img
        if len(label) > 1: # get unicode code of letter
            code = str(ord(label[0])) + '_' + str(ord(label[1]))
        else: code = str(ord(label))
        try: # check if there's a folder for this letter
            os.chdir(join(destpath,code+label))
        except: # if not, make one
            os.mkdir(join(destpath,code+label))
            os.chdir(join(destpath,code+label))
        # move letter image to its appropriate folder
        os.rename(join(datapath,img), join(destpath,code+label,img))
