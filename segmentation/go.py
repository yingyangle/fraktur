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


# path to a book folder -- use if you're only running seg on one book
folder_path = your_path_here+'/data/bravo'
# path to folder of book folders -- use ise you're running on folder of books
all_folders_path = join(your_path_here, 'data')
# list of folders in all_folders_path that are not book folders
ignore_folders = ['letter_data', 'letter_data1', 'dataset']

# *** MAKE SURE THE FOLLOWING 2 FOLDERS/PATHS EXIST *** #
# path to letters_data folder -- where each letter folder will be stored
output_path = join(your_path_here, 'data/letter_data')
# path to segs folder -- where contour images will be saved for each book
# unimportant, mostly for testing/debugging seg.py
segs_path = join(your_path_here, 'segmentation/segs')

os.chdir(output_path)
aus = open('errors.txt', 'w') # error log
aus.close()


# run letter segmentation on all line images in a folder
def segFolder(folder_path):
    i = 0 # for testing purposes
    folder_name = folder_path[folder_path.rfind('/')+1:] # name of book folder
    segs_folder_path = join(segs_path, folder_name) # path to save contour imgs for this book
    # folder in output_path to store letter folders for this book
    try: os.mkdir(join(output_path, folder_name))
    except FileExistsError: pass
    # for each line img in this book folder
    for img in [x for x in os.listdir(folder_path) if x[-3:] == 'png']:
        # make words folder to store word-segmented images
        try: os.mkdir(join(output_path, 'words'))
        except FileExistsError:
            shutil.rmtree(join(output_path, 'words'))
            os.mkdir(join(output_path, 'words'))
        words_path = join(output_path, 'words') # path to words folder
        # get word-segmented images
        try: wordSeg(img, folder_path, words_path)
        except: # write to error log if fail
            aus = open('errors.txt', 'a')
            aus.write('Failed seg: '+folder_path+' '+img+'\n')
            aus.close()
        print('finished wordSeg:', img, '~~~~~~~~~~~~~~~~~~~~~~~~~')
        # for each word img in this line
        for word_img in [x for x in os.listdir(words_path) if x[-3:] == 'png']:
            # segment letters in this word
            try: seg(word_img, 1, words_path, segs_folder_path, folder=folder_name)
            except: # write to error log if fail
                aus = open('errors.txt', 'a')
                aus.write('Failed seg: '+folder_path+' '+word_img+'\n')
                aus.close()
        shutil.rmtree(words_path) # delete words folder, clear it for next line
        if i % 50 is 0:
            # separate letter imgs into folders
            print('Separating data into folders...')
            labelData(segs_folder_path, join(output_path, folder_name))
        i += 1
        # if i > 3: break # for testing purposes
    # separate letter imgs into folders
    print('Separating data into folders...')
    labelData(segs_folder_path, join(output_path, folder_name))
    return


### execute ###

os.chdir(all_folders_path) # path to folder of book folders
# get list of book folders
folders = [x for x in os.listdir(all_folders_path) if os.path.isdir(join(all_folders_path, x))]
for f in ignore_folders: # remove folders we don't wanna check
    try: folders.remove(f)
    except: print('failed to remove', f)

# for each book folder in all_folders_path
for foldername in folders:
    folder_path = join(all_folders_path, foldername) # path to img/txt data for this book
    segFolder(folder_path)
