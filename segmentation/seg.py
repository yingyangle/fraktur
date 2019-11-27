# Christine Yang
# Fraktur Cracker
# seg.py
# attempt at simple character segmentation

import os, re, glob, codecs, more_itertools as mit
from PIL import Image

W = 0 # how many non-whitespace pixels to allow and still count it as whitespace
os.chdir('/Users/Christine/cs/fraktur')
main_dir = os.getcwd()

# write list of unique chars to .txt file
def getAlphabet():
    aus = codecs.open('../char_set.txt', 'w', 'utf-8')
    aus.write('Number of Unique Chars: ' + str(len(chars_list)))
    a = [aus.write(x+'\n') for x in chars_list]
    aus.close()

# get correct letter transcriptions for this line of text
def getLabels(filename):
    txt_file = filename[:-4]+'.txt' # get .txt filename
    ein = open(txt_file, 'r') # open .txt file
    raw = ein.read().rstrip() # read .txt file
    ein.close()
    txt = re.sub('[.,\'\"“„ ]', '', raw) # replace spaces and punctuation
    # add # chars for extra letters at the end from bad segmentation
    return txt+'{0:#^50}'.format('')

# (try to) split image line of text into individual letters, one image per letter
def getLetters(filename):
    os.chdir(main_dir+'/segmentation/test_data')
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
        os.mkdir('../letters/man')
        os.chdir('../letters/man')
    except: os.chdir('../letters/man')

    for i in range(len(groups)-1): # crop letter images
        left = groups[i][-1]
        right = groups[i+1][0]
        area = (left, 0, right, height)
        cropped_im = im.crop(area)
        # cropped_im.show()
        imagename = filename[:-4]+'_'+str(i)+'_'+labels[i]+'.png'
        cropped_im.save(imagename)
    os.chdir('..')

# get set of unique chars appearing in txt files for a book
def getAlphabet(foldernames):
    chars_set = set() # set of unique chars
    if len(foldernames) > 1: # if we get a list of folders
        [chars_set.update(getAlphabet([x])) for x in foldernames]
    elif len(foldernames) == 1: # if we get only 1 folder
        os.chdir(main_dir+'/data/'+foldernames[0])
        for file in glob.glob("*.txt"): # for each .txt file in this folder
            ein = codecs.open(file, 'r', 'utf-8')
            raw = ein.read()
            chars_set.update(raw)
            ein.close()
        return chars_set
    # write results
    chars_list = list(chars_set)
    chars_list.sort()
    aus = codecs.open(main_dir+'/char_set.txt', 'w', 'utf-8')
    aus.write('Number of Unique Chars: ' + str(len(chars_list)))
    temp = [aus.write(x+'\n') for x in chars_list]
    aus.close()
    return chars_list


### execute ###

# segment letters

os.chdir(main_dir+'/segmentation/test_data')
images1 = ['a.png', 'b.png', 'hi.png']
images2 = ['hard.png', 'hard2.png', 'hoff.png']

for img in images1: getLetters(img)

# get set of unique chars

# os.chdir(main_dir+'/data')
# # folders = ['1797-wackenroder_herzensergiessungen', '1853-rosenkranz_aesthetik']
# folders = [x for x in os.listdir() if x[0] != '.']
#
# chars_list = getAlphabet(folders)
