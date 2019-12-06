# Christine Yang
# Fraktur Cracker
# wordSeg.py
# word segmentation

import os, numpy as np, cv2, re, more_itertools as mit, shutil
from PIL import Image
from preprocess import preprocess

# get correct labels from .txt transcription
def getLabels(filename):
    txt_file = filename[:-4]+'.txt' # get .txt filename
    ein = open(txt_file, 'r') # open .txt file
    raw = ein.read().rstrip() # read .txt file
    ein.close()
    txt = re.sub('[.,\'\"“„-]', '', raw) # replace punctuation
    txt = re.sub(r'\\', r'\\', txt)
    # add '#' chars for extra letters at the end from bad segmentation
    filler = ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'] # len=13
    word_ls = txt.split(' ') + filler
    return word_ls # list of letter labels

# segment line img into word imgs
def wordSeg(filename, datapath, destpath):
    print(filename)
    os.chdir(datapath) # path of original line img and label .txt
    labels = getLabels(filename)
    im = Image.open(filename, 'r') # open image
    width, height = im.size # image size
    pix_val = list(im.getdata()) # pixel color values
    cols = np.array([pix_val[n::width] for n in range(width)]) # pixels as columns
    half_cols = cols[:,0:len(cols[0])//5*3]
    # get column indices of whitespace
    whitespace = []
    for i in range(len(half_cols)):
        c = half_cols[i]
        whites = [x for x in c if x >= 200] # whitespace pixels
        if len(whites) >= len(c): whitespace.append(i)
    # group consecutive whitespace
    groups = [list(group) for group in mit.consecutive_groups(whitespace)]
    groups_len = [len(g) for g in groups]
    # get indices in groups of longest consecutive whitespace
    num_spaces = len(labels) - 1 - 13 # minus length of filler '#'s
    spaces_i = sorted(range(len(groups_len)), key = lambda sub: groups_len[sub])[-num_spaces:]
    spaces_i.sort()
    # top N longest consecutive whitespace, where N is # spaces in the line
    spaces = [groups[i] for i in spaces_i]
    spaces = [[0]] + spaces + [[len(cols)]]
    os.chdir(destpath) # destination to save word imgs
    for i in range(len(spaces)-1):
        left = spaces[i][-1]
        right = spaces[i+1][0]
        area = (left, 0, right, height)
        cropped_im = im.crop(area)
        # imagename = linenum_wordnum_word.png
        imagename = filename[:-4]+'_'+str(i)+'_'+labels[i]+'.png'
        imagename = re.sub('/', 'sl', imagename)
        cropped_im.save(imagename)
    return


### execute / testing ###

# os.chdir('/Users/ovoowo/Desktop/fraktur/segmentation/test_data')
# os.chdir('/Users/Christine/Documents/cs/fraktur/segmentation/test_data')
# try: shutil.rmtree('test')
# except FileNotFoundError: pass
# os.mkdir('test')
# filename = 'sane.png'
# wordSeg(filename, os.getcwd(), os.getcwd()+'/test')
