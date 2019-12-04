# Christine Yang
# Fraktur Cracker
# go.py
# run wordSeg.py + seg.py on folders in fraktur/data, then organize with labelData.py

import os, shutil
from os.path import join
from seg import seg
from wordSeg import wordSeg
from labelData import labelData

# your_path_here = '/Users/ovoowo/Desktop/fraktur'
your_path_here = '/Users/Christine/Documents/cs/fraktur'

########## TO RUN ON FULL DATASET: ##########
# - we'll only use RIDGES-Fraktur and dta19 in GT4HistOCR/corpus
#
# - run renameFiles.py setting your_path to the path to RIDGES-Fraktur
# - run go.py, but replace the path in the line with "### HERE ###" to the path to RIDGES-Fraktur
# - run checkLabels.py replacing letter_data with the path to the letter_data folder in RIDGES-Fraktur
# - repeat the above 3 steps for dta19

# path to a book folder -- use if you're only running seg on one book
folder_path = your_path_here+'/data/bravo'
# path to folder of book folders -- use ise you're running on folder of books
all_folders_path = join(your_path_here, 'data')
# list of folders in all_folders_path that are not book folders
ignore_folders = ['letter_data', 'letter_data1', 'dataset']

# *** MAKE SURE THE FOLLOWING 2 FOLDERS/PATHS EXIST ***
# path to letters_data folder -- where each letter folder will be stored
letters_path = join(your_path_here, 'data/letter_data')
# path to segs folder -- where contour images will be saved for each book
# unimportant, mostly for testing/debugging seg.py
segs_path = join(your_path_here, 'segmentation/segs')


# run letter segmentation on all line images in a folder
def segFolder(folder_path):
    stopp = 0 # for testing purposes
    folder_name = folder_path[folder_path.rfind('/')+1:] # name of book folder
    segs_folder_path = join(segs_path, folder_name) # path to save contour imgs for this book
    # for each line img in this book folder
    for img in [x for x in os.listdir(folder_path) if x[-3:] == 'png']:
        # make words folder to store word-segmented images
        try: os.mkdir(join(letters_path, 'words'))
        except FileExistsError:
            shutil.rmtree(join(letters_path, 'words'))
            os.mkdir(join(letters_path, 'words'))
        words_path = join(letters_path, 'words') # path to words folder
        # get word-segmented images
        print('finished wordSeg:', img, '~~~~~~~~~~~~~~~~~~~~~~~~~')
        wordSeg(img, folder_path, words_path)

        # for each word img in this line
        for word_img in [x for x in os.listdir(words_path) if x[-3:] == 'png']:
            # segment letters in this word
            seg(word_img, 1, words_path, segs_folder_path)
        shutil.rmtree(words_path) # delete words folder, clear it for next line
        stopp += 1
        if stopp > 3: break # for testing purposes
    # separate letter imgs into folders
    print('Separating data into folders...')
    labelData(segs_folder_path, letters_path)
    return


### execute ###

os.chdir(all_folders_path) # path to folder of book folders
# get list of book folders
folders = [x for x in os.listdir(all_folders_path) if os.path.isdir(join(all_folders_path, x))]
for f in ignore_folders: # remove folders we don't wanna check
    try: folders.remove(f)
    except: print('failed to remove', f)
print(all_folders_path)
print(folders)

# for each book folder in all_folders_path
for foldername in folders:
    print('started folder', foldername)
    folder_path = join(all_folders_path, foldername) # path to img/txt data for this book
    segFolder(folder_path)
