#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:27:08 2019

@author: ovoowo
"""
# # # # # # # # # # # # # #
# Yuezhen Chen
# Please Work

import os,numpy as np,shutil
from relabel import reLabel
from KCluster import silhouette, kMeanclf,fetchFeature

your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
datapath = your_path_here+'fraktur/features/'
storepath = your_path_here +'fraktur/cleaning/' #the place to store info for cleaning
folder = 'testdata/letter_data/' #the one store imgs we want to clean
imgpath =your_path_here +'fraktur/'+folder #the place to get all books imgs'feature
txtpath = storepath+'letterfeatures/' #to get the error and feature txt for books

###First called dataforKMeans once in features folder to get all the unlabel features
#get all the folder's name that need to be cleaned
os.chdir(txtpath)
books = os.listdir(txtpath)[1:]
tracker = 0
books.remove('0')
books.remove('no')
for book in books:
    infopath = txtpath+book+'/'
    all = os.listdir(infopath)
    btxts = [x for x in all if x[-5:] == 'b.txt']
    dtxts = [x for x in all if x[-5:] == 'd.txt']
    etxts = [x for x in all if x[-5:] == 's.txt']
    totalletter = len(btxts)
    lettertracker = len(btxts)
    for i in range(len(btxts)): #for each letter
        foldername = btxts[i][:-7]
        print('foldername=',foldername)
        bX = fetchFeature(btxts[i],infopath)
        dX = fetchFeature(dtxts[i],infopath)
        bnumK = silhouette(bX)
        dnumK = silhouette(dX)
        print('the number of cluster for blackness =', bnumK, '\nthe number of cluster for distance = ', dnumK)
        (idbLabel,blabels) = kMeanclf(bX,bnumK)
        (iddLabel,dlabels) = kMeanclf(dX,dnumK)
        print('For blackness, the largest cluster has label: ', idbLabel)
        print('For distance, the largest cluster has label: ', iddLabel)
        os.chdir(storepath)
        try:
            os.mkdir(foldername)#create store folder for each letter
        except:
            print(foldername,' folder created')
        letterpath = imgpath + book + '/' + foldername+'/' #to access latter images
        destpath = storepath+foldername+'/'
        if foldername+'errors.txt' in etxts:
            reLabel(infopath,blabels,dlabels,foldername,letterpath) #foldername
        else:
            reLabel('noerror',blabels,dlabels,foldername,letterpath)
        imgs = np.array([x for x in os.listdir(letterpath) if x[-6:] != '{}{}.png'.format(idbLabel,iddLabel)]) #or x[:3] =='###'
        total = imgs.size
        counter = 0
        for img in imgs:
            shutil.move(img, destpath+'/'+img)#copy every image to the dataset folder
            counter += 1
            if total < 5000 and counter%100 == 99:
                print(str(counter)+' bad images/ '+str(total)+' images moved\t')
            if total >=5000 and counter%500 == 499:
                print(str(counter)+' bad images/ '+str(total)+' images moved\t')
            print(str(counter)+' bad images/ '+str(total)+' images moved\t')
        gimgs = np.array([x for x in os.listdir(letterpath)])
        os.chdir(letterpath)
        for gimg in gimgs:
            os.rename(gimg,gimg[:-6]+gimg[-4:]) #get rid of the label for getFeature to extract label
        lettertracker -= 1
        print('='*40+'\n'+str(lettertracker)+'out of '+str(totalletter)+' left for CLEANING\n'+'='*40)

    tracker += 1
    print('*'*40+'\n'+str(len(books)-tracker)+' left for CLEANING\n'+'*'*40)
