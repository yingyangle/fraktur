# Christine Yang
# Fraktur Cracker
# seg.py
# attempt at simple character segmentation

import os, re, glob, codecs, more_itertools as mit
from PIL import Image

W = 0 # how many non-whitespace pixels to allow and still count it as whitespace
os.chdir('/Users/Christine/cs/fraktur/segmentation')
main_dir = os.getcwd()

# get correct letter transcriptions for this line of text
def getLabels(filename):
    ein = open(filename)
    txt = ein.read()[:-1]
    ein.close()
    return re.sub(' ', '', txt) + '#############################################'

# (try to) split image line of text into individual letters, one image per letter
def getLetters(filename):
    labels = getLabels(filename) # correct transcription
    im = Image.open(filename, 'r') # open image
    width, height = im.size # image size
    pix_val = list(im.getdata()) # pixel color values

    cols = [pix_val[n::width] for n in range(width)] # pixels as columns
    # print('\n'.join(' '.join(map(str,sl)) for sl in cols)) # print cols

    # get column indices of whitespace
    whitespace = []
    for i in range(len(cols)):
        c = cols[i]
        whites = [x for x in c if x >= 200] # whitespace pixels
        if len(whites) + W >= len(c): whitespace.append(i) # allow W non-whitespace
        # all(x >= 200 for x in c)

    # group consecutive whitespace
    groups = [list(group) for group in mit.consecutive_groups(whitespace)]

    try: # folder for cropped letter images
        os.mkdir('letters')
        os.chdir('letters')
    except: os.chdir('letters')

    for i in range(len(groups)-1): # crop letter images
        left = groups[i][-1]
        right = groups[i+1][0]
        area = (left, 0, right, height)
        cropped_im = im.crop(area)
        # cropped_im.show()
        imagename = filename[:-4]+' '+str(i)+' '+labels[i]+'.png'
        cropped_im.save(imagename)
    os.chdir('..')

# getLetters('hi.png')
# getLetters('a.png')
# getLetters('b.png')
# getLetters('hard.png')
# getLetters('hard2.png')



# get set of unique chars appearing in txt files for a book
def getAlphabet(foldername):
    os.chdir('../data')
    chars_set = set() # set of unique chars
    for file in glob.glob("*.txt"): # for each .txt file in this folder
        ein = codecs.open(file,'r', 'utf-8')
        raw = ein.read()
        chars_set.update(raw)
        ein.close()
    os.chdir(main_dir)
    return chars_set

# get set of unique chars
chars_set = getAlphabet('1797-wackenroder_herzensergiessungen')
chars_set.union(getAlphabet('1853-rosenkranz_aesthetik'))
chars_list = list(chars_set)
chars_list.sort()


# write list of unique chars to .txt file
aus = codecs.open('../char_set.txt', 'w', 'utf-8')
aus.write('Number of Unique Chars: ' + str(len(chars_list)))
a = [aus.write(x+'\n') for x in chars_list]
aus.close()
