# Christine Yang
# Fraktur Cracker
# go.py
# run wordSeg.py + seg.py on folders in fraktur/data, then organize with labelData.py

import os, shutil
from seg import seg
from wordSeg import wordSeg
from labelData import labelData

# your_path_here = '/Users/ovoowo/Desktop/fraktur'
your_path_here = '/Users/Christine/cs/fraktur'
seg_dir = your_path_here + '/segmentation'

stopp = 0
# list of book folders in 'data'
os.chdir(your_path_here + '/data')
folders = [x for x in os.listdir() if os.path.isdir(os.path.join(os.getcwd(), x))]
folders.remove('letter_data')
for foldername in folders: # for each book folder in 'data'
    datapath = your_path_here+'/data/'+foldername # path to img/txt data for this book
    try: # make folder in 'segmentation/segs' to store morph imgs for this book
        os.mkdir(seg_dir+'/segs/'+foldername)
    except FileExistsError: pass
    destpath = seg_dir+'/segs/'+foldername # path to save segmented letter imgs for this book
    for img in [x for x in os.listdir(datapath) if x[-3:] == 'png']: # for each line img in this book/folder
        try: os.mkdir(datapath+'/temp') # to store word images
        except FileExistsError: pass
        wordspath = datapath+'/temp' # to store word images
        wordSeg(img, datapath, wordspath) # get word segmented images
        print('wordSeg', img, '~~~~~~~~~~~~~~~~~~~~~~~~~')
        for wordImg in [x for x in os.listdir(wordspath) if x[-3:] == 'png']: # for each word img in this line
            print('seg', wordImg)
            seg(wordImg, 1, wordspath, destpath) # segment letters
        shutil.rmtree(wordspath) # delete 'temp' folder
        stopp += 1
        if stopp > 8: break
    labelData(destpath, your_path_here+'/data/letter_data/') # separate letter imgs into folders
