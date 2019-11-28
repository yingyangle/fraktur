# Christine Yang
# Fraktur Cracker
# labelData.py
# separates labeled data into folders for each letter

import os, re

os.chdir('/Users/Christine/cs/fraktur/segmentation/letters/')
main = os.getcwd()

# each letter .png in folder (don't include filename_morph.png and filename.png)
images = [x for x in os.listdir() if x[-3:] == 'png' and x.count('_')>1]

# for each letter image in the folder
for img in images:
    index = img.rfind('_') # index of last occurrence of '_' in filename
    label = img[index+1:-4] # letter label of this img
    if len(label) > 1:
        code = str(ord(label[0])) + '_' + str(ord(label[1]))
    else: code = str(ord(label))
    try: # check if there's a folder for this letter
        os.chdir(main+'/data/'+code+label)
    except: # if not, make one
        os.mkdir(main+'/data/'+code+label)
        os.chdir(main+'/data/'+code+label)
    # move letter image to its appropriate folder
    os.rename(main+'/'+img, main+'/data/'+code+label+'/'+img)
