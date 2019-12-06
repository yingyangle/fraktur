# Christine Yang
# Fraktur Cracker
# go.py
# run wordSeg.py + seg.py on folders in fraktur/data, then organize with labelData.py

import os, shutil
from seg import seg
from wordSeg import wordSeg
from labelData import labelData

your_path_here = '/Users/ovoowo/Desktop'
#your_path_here = '/Users/Christine/cs/fraktur'
folderpath = your_path_here+'/GT4HistOCR/corpus/RIDGES-Fraktur/'
folder ='1722-FloraSaturnizans-Henckel'
seg_dir = your_path_here + '/fraktur/testdata/'

########## TO RUN ON FULL DATASET: ##########
# - we'll only use RIDGES-Fraktur and dta19 in GT4HistOCR/corpus
#
# - run renameFiles.py setting your_path to the path to RIDGES-Fraktur
# - run go.py, but replace the path in the line with "### HERE ###" to the path to RIDGES-Fraktur
# - run checkLabels.py replacing letter_data with the path to the letter_data folder in RIDGES-Fraktur
# - repeat the above 3 steps for dta19


stopp = 0
# list of book folders in 'data'
os.chdir(folderpath) ### HERE ### !!!!!!!!!!!
folders = [x for x in os.listdir() if os.path.isdir(os.path.join(os.getcwd(), x))]
#folders.remove('letter_data')
#for foldername in folders: # for each book folder in 'data'
datapath = folderpath+folder # path to img/txt data for this book
try: # make folder in 'segmentation/segs' to store morph imgs for this book
    os.mkdir(seg_dir+foldername)
except FileExistsError: pass
destpath = seg_dir+foldername # path to save segmented letter imgs for this book
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
labelData(destpath, your_path_here+'/fraktur/testdata/letter_data/') # separate letter imgs into folders
