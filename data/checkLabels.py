# Christine Yang
# Fraktur Cracker
# checkLabels.py
# check for correct labels on segmented letter images in data/letter_data
# move incorrectly labeled images to 'bad' folder

import os

# your_path_here = '/Users/ovoowo/Desktop/fraktur'
your_path_here = '/Users/Christine/cs/fraktur'
letter_data = your_path_here + '/data/letter_data/'

os.chdir(letter_data) # folder containing folders for each letter label
try: os.mkdir('bad') # folder to store incorrectly labeled images
except FileExistsError: pass

umlaut = 'ü'[-1]

# get list of letter labels from string of letter labels
def getLabels(chars_str):
    chars_ls = [] # chars_str in list form, with digraphs and problem diacritics joined together
    i = 0
    while i < len(chars_str): # for each char in chars_str
        c = chars_str[i]
        if chars_str[i] == 'c' and chars_str[i+1] == 'h':
            chars_ls.append('ch') # treat 'ch' as one char
            i += 1
        elif chars_str[i] == 'c' and chars_str[i+1] == 'k':
            chars_ls.append('ck') # treat 'ch' as one char
            i += 1
        elif chars_str[i] == 'f' and chars_str[i+1] == 'f':
            chars_ls.append('ff') # treat 'ff' as one char
            i += 1
        elif chars_str[i] == 'ſ' and chars_str[i+1] == 'ſ':
            chars_ls.append('ſſ') # treat 'ſſ' as one char
            i += 1
        elif chars_str[i] == 'l' and chars_str[i+1] == 'l':
                    chars_ls.append('ll') # treat 'll' as one char
                    i += 1
        elif chars_str[i] == 'ͤ': # treat this as diacritic, not separate letter
            chars_ls = chars_ls[:-1]
            chars_ls.append(chars_str[i-1]+chars_str[i])
        elif chars_str[i] == umlaut: # treat this as diacritic, not separate letter
            chars_ls = chars_ls[:-1]
            chars_ls.append(chars_str[i-1]+chars_str[i])
        else: chars_ls.append(chars_str[i])
        i += 1
    return chars_ls # list of letter labels

# for each letter folder
for folder in [x for x in os.listdir() if x != '35#' and x != 'bad' and x[0] != '.']: 
    os.chdir(letter_data+folder)
    if len(os.listdir()) is 0: continue
    # the letter label for all imgs in this folder
    letter = os.listdir()[0][:-4].split('_')[-1] 
    goods = 0 # num correctly labeled images in this folder
    bads = 0 # num incorrectly labeled images in this folder
    
    for img in os.listdir(): # for each letter image
        parts = img[:-4].split('_')
        word = getLabels(parts[2]) # the word that this letter is in
        index = int(parts[3]) # index of letter in word
        if len(letter) == 1:
            if word[index] == letter: # if letters match, it's good
                goods += 1
            else: # if letters don't match, move it to bad folder
                os.rename(letter_data+folder+'/'+img, letter_data+'bad/'+img)
                bads += 1
        else:
            if word[index:index+2] == letter # if letters match, it's good
                goods += 1
            else: # if letters don't match, move it to bad folder
                os.rename(letter_data+folder+'/'+img, letter_data+'bad/'+img)
                bads += 1
    print(letter, ':', str(goods), '/', str(goods+bads), 'matched')