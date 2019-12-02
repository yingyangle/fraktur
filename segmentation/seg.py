# Christine Yang
# Fraktur Cracker
# seg.py
# third attempt at character segmentation, using OpenCV

# OPENCV
# https://stackoverflow.com/questions/49291770/ocr-cropping-the-letters
# https://stackoverflow.com/questions/42316315/how-to-segment-license-plate-characters-removing-unwanted-characters-using-openc
# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
# https://circuitdigest.com/tutorial/image-segmentation-using-opencv
# https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
# https://stackoverflow.com/questions/50432349/combine-contours-vertically-and-get-convex-hull-opencv-python
# ROI = region of interest

# thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

import os, codecs, numpy as np, cv2, re, sys, shutil
from preprocess import preprocess

# letters than tend to get divorced
divorced = ['u', 'ů', 'ü', 'ù', 'û', 'n', 'm', 'w', 'h', 'U', 'Ü', 'Ù', 'Ú', 'Û']
# letters that tend to blend into the next letter
blenders = ['e', 'è', 'é', 'ê', 'ë', 'r', 'v', 'ſ', 't']
umlaut = 'ü'[-1]
np.set_printoptions(threshold=sys.maxsize) # print full np arrays untruncated, for testing

# get correct labels from .txt transcription
# type=0 means we're given a line of text, type=1 means we're given a word
def getLabels(filename, type):
    txt = ''
    if type is 0: # non word-segmented imgs
        txt_file = filename[:-4]+'.txt' # get .txt filename
        ein = open(txt_file, 'r') # open .txt file
        txt = ein.read().rstrip() # read .txt file
        ein.close()
    elif type is 1: # word-segmented imgs
        index = filename.rfind('_') # index of last occurrence of '_' in filename
        txt = filename[index+1:-4] # word label of this img
    txt = re.sub('[\.\,\'\"\“\„\-]', '', txt) # replace punctuation
    txt = re.sub(' ', '', txt) # replace spaces
    # add '#' chars for extra letters at the end from bad segmentation
    chars_str = txt+'{0:#^50}'.format('')
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

# get binary img matrix with just the contour filled in
# inv=0 for non-inverted nimg, inv=1 for inverted nimg
def getContour(nimg, contour, inv):
    if inv == 0: # for non-inverted img
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, [contour], 0, 0, -1) # color contour area as 0
        out = np.full_like(nimg, 255) # create array of 255s (whites)
        out[mask == 0] = nimg[mask == 0] # where mask is 0, change out value to img value
        x, y, w, h = cv2.boundingRect(contour) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting boxed roi (region of interest)
    else: # for inverted img
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, [contour], 0, 0, -1) # color contour area as 0
        out = np.full_like(nimg, 0) # create array of 0s (blacks)
        out[mask == 0] = 255 # where mask is 0, change out value to img value
        x, y, w, h = cv2.boundingRect(contour) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting boxed roi (region of interest)
    return roi

# takes original list of contours and binary image
# returns binary image with spaces filled in
def fillInSpaces(bin, contours):
    out = np.full_like(bin, 0) # create array of 0s (blacks)
    for c in contours:
        ones = np.ones_like(bin) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, [c], 0, 0, -1) # color contour area as 0
        out[mask == 0] = 255 # where mask is 0, change out value to img value
    return out

# drip black pixels down or sideways to connect diacritics or gaps within a single letter
# takes binary inverted img, orig binary inverted img, contour dimensions, and drip direction
# returns new binary inverted img with drip added
# dir: 0 = drip down, 1 = drip right
# we need orig binary inverted img to find stop point without previous drips drawn on
def drip(bin, bin_orig, dims, dir):
    x, y, w, h = dims # dimensions of contour bounding box
    if dir == 0: # drip down
        xx = x+(w//2) # x center index (of contour)
        yy = y+(h//2) # y center index (of contour)
        try: # drip down from center
            # index of first 255 pixel going down from xx,yy, aka stopping point
            stop = y+h + np.where(bin_orig[:,xx][y+h+1:]==255)[0][0]
            # replace 0s in bin with 255s from contour center down to first 0
            bin[:,xx][yy:stop+1] = 255
        except IndexError: # if no black pixels in center col to stop on,
            try: # drip down from left edge
                # index of first 255 pixel going down from xx,yy, aka stopping point
                stop = y+h + np.where(bin_orig[:,x][y+h+1:]==255)[0][0]
                # replace 0s in bin with 255s from contour left edge down to first 0
                bin[:,x][yy:stop+1] = 255
            except IndexError: # if no black pixels in left edge col either,
                stop = len(bin_orig) # drip all the way down
                bin[:,xx][yy:stop+1] = 255 # from the center of the contour
    if dir == 1: # drip right
        xx = x+(w//2) # x center index (of contour)
        yy = y+(h//2) # y center index (of contour)
        # index of first 255 pixel going right from xx,yy, aka stopping point
        try: stop = x+w + np.where(bin_orig[yy][x+w+1:]==255)[0][0]
        except: stop = len(bin_orig[0])
        # replace 0s in bin with 255s from contour center down to first 0
        bin[yy][xx:stop+1] = 255
    return bin # return new binary inverted img with drip added

# check for divorced u's, n's, m's, and w's
# take a single char img + label, return result
def sanityCheckDivorce(bin, bin_orig, dims, lab):
    x, y, w, h = dims # dimensions of contour bounding box
    ratioThresh = -1 # width/height ratio can't be less than this
    if lab in ['m', 'w']: ratioThresh = 1 # set ratioThresh
    elif lab is 'h': ratioThresh = 0.35
    elif lab is 'u' or lab is 'n': ratioThresh = 0.7
    else: ratioThresh = 0.5 # set ratioThresh for n's and other u's
    ratio = w / h # width/height ratio of char image
    change = 0
    if ratio < ratioThresh: # if below ratioThresh, drip right
        bin = drip(bin, bin_orig, dims, 1)
        change = 1
    return (bin, change)

# check for e's, r's, v's, and ſ's blending into next letter
def sanityCheckBlend(bin, lab, contour):
    temp = getContour(bin, contour, 1) # char matrix in bounding box
    x, y, w, h = cv2.boundingRect(contour) # dimensions of contour bounding box
    ratioThresh = 100 # width/height ratio can't be more than this
    if lab in ['e', 'r', 't']: ratioThresh = 0.85
    if lab in ['v']: ratioThresh = 1
    ratio = w / h # width/height ratio of char image
    change = 1
    if lab == 'ſ':
        bottom_right = temp[h//2:][w//2:] # bottom right quarter of img
        top_right = temp[h//20:h//2][:,w//2:] # top right quarter with top 5% cut off
        blacks = [np.count_nonzero(x) for x in bottom_right] # count black pixels per row
        if sum(blacks) > bottom_right.size*0.2: # if a good number of blacks in this area,
            blacks = np.array([np.count_nonzero(x) for x in top_right])
            blacks = np.where(blacks==0, 999, blacks)
            # bin index of row in this char with fewest black pixels
            boundary_i = np.ma.array(blacks).argmin() + y + h//20
            bin[boundary_i][x+w//2:x+w] = 0
        else: change = 0
    elif ratio > ratioThresh: # if above ratioThresh, split it up
        mid = temp[:,(w//10):(w-w//10)] # cut off the left and right 10% of the char img
        blacks = [np.count_nonzero(x) for x in mid.T] # count black pixels per col
        # bin index of col in this char with fewest black pixels
        boundary_i = np.array(blacks).argmin() + x + w//10
        bin[:,boundary_i] = 0
    else: change = 0
    return (bin, change)

# check for trash contours, return 1 if trash, 0 if not
def checkTrash(contours, nimg, bin):
    new_contours = []
    for i in range(len(contours)): # remove trash contours
        x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box
        # ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        # mask = cv2.drawContours(ones, contours, i, 0, -1) # color contour area as 0
        # if it's really small, it's probably noise
        if h < len(nimg)/11: continue
        # if fairly small and in bottom half of image, probably noise/punctuation
        if h < len(nimg)*0.4:
            if y >= len(nimg)/2: continue
        # white_pct = np.count_nonzero(bin[mask == 0]==0) / len(bin[mask == 0])
        # print(i, np.count_nonzero(bin[mask == 0]==0), len(bin[mask == 0]), white_pct)
        # # if 60% of region is white, don't count it (prob inside of 'e', 'd', etc.)
        # if white_pct > 0.5: continue
        # # if 40% region is white and region is small, don't count it
        # if len(bin[mask == 0]) < len(nimg):
        #     if white_pct > 0.3: continue
        new_contours.append(contours[i]) # if passes all checks, add to new_contours
    return new_contours

# perform diacritic checks and sanity checks and morph img accordingly
# return new binary image
# sanity indicates whether or not to perform sanity checks
def morph(filename, type, destpath, sanity):
    labels = getLabels(filename, type) # get correct labels
    img, nimg, bin = preprocess(filename) # get img, flat img, and binary inverted img
    bin_orig = bin.copy()
    # find contour regions
    _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort contours in order
    bin = fillInSpaces(bin, contours) # fill in white spaces within letters
    _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort contours in order
    ii = 0 # index of letter in labels
    i = 0 # index for contour regions
    next_letter = 1 # how much to increment index to next letter in txt labels
    diacritic_flag = 0 # whether we performed a diacritic morph on this letter
    divorce_flag = 0 # whether we're currently on a divorced letter
    blender_flag = 0 # whether we performed a blender morph on this letter
    contours = checkTrash(contours, nimg, bin) # get rid of trash contours
    # for i in range(len(contours)): # save contours for testing/debugging
    #     x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box for cropping
    #     cv2.imwrite('temp/'+filename+str(i)+'_'+str(sanity)+'.png',bin[y:y+h, x:x+w])
    while i < len(contours): # for each contour
        x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box for cropping
        dims = [x, y, w, h] # dimensions of bounding box in list form
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, contours, i, 0, -1) # color contour area as 0
        lab = labels[ii] # get letter label for current contour
        ### DIACRITIC CHECK ###
        if h < len(nimg)/3: # if region really small, might be noise
            if y+h < len(nimg)/2: # but if it's in the top half, probably diacritic
                bin = drip(bin, bin_orig, dims, 0) # drip down to connect it to letter body
                lab = labels[ii-1] # use prev letter label, since diacritics usually after letter body
                diacritic_flag = 1 # stay on current letter which is actually the next one
        elif sanity is 1: # if we wanna perform sanity checks
            ### SANITY CHECK for DIVORCEES ###
            if lab in divorced: # check for commonly divorced letters
                if divorce_flag != 1: # =1 means connection already drawn from other half of divorced chunk
                    bin, divorce_flag = sanityCheckDivorce(bin, bin_orig, dims, lab)
                else: divorce_flag = 0 # exit divorce_flag
            ### SANITY CHECK for BLENDING ###
            elif lab in blenders: # check for commonly blended letters
                if lab == 'ſ':
                    if labels[ii+1] == 't':
                        bin, blender_flag = sanityCheckBlend(bin, lab, contours[i])
                        print(blender_flag)
                else: bin, blender_flag = sanityCheckBlend(bin, lab, contours[i])
            # make sure labels matched correctly, otherwise don't do sanity checks
            if lab == '#':
                cv2.imwrite(destpath+'/###'+filename[:-4]+'_SANITY.png', bin)
                return morph(filename, type, destpath, 0)
            if i is len(contours)-1 and labels[ii+1] != '#':
                if blender_flag != 1:
                    cv2.imwrite(destpath+'/###'+filename[:-4]+'_SANITY.png', bin)
                    return morph(filename, type, destpath, 0)
        # determine next_letter increment
        if diacritic_flag == 1: # if this contour was a diacritic mark,
            next_letter = 0 # we're already on the next letter, so stay on it
            diacritic_flag = 0 # reset diacritic_flag
        elif divorce_flag == 1: # if currently on divorced letter,
            next_letter = 0 # don't advance to next letter until divorce is settled
        elif blender_flag == 1:
            next_letter = 2
            blender_flag = 0
        else: next_letter = 1 # otherwise, go on to next letter label
        ii += next_letter # move on to next letter in labels
        i += 1 # move on to next contour
    # save resulting binary img after morphological operations
    if sanity == 0: filename = '###' + filename
    cv2.imwrite(destpath+'/'+filename[:-4]+'_morph.png', bin)
    return (img, nimg, bin, sanity)

# save image for each segmented char, and full image with char boundaries drawn on
# type=0 means we're given a line of text, type=1 means we're given a word
def seg(filename, type, datapath, destpath, thck=2):
    os.chdir(datapath)
    # orig img, flat img, and binary inverted img adjusted for diacritic and sanity checks
    img, nimg, bin, sanity = morph(filename, type, destpath, 1)
    labels = getLabels(filename, type) # get correct labels
    if sanity == 0: filename = '###' + filename
    os.chdir(destpath) # dir to save cropped letter images
    _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contour regions
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours = checkTrash(contours, nimg, bin)
    ii = 0 # index of letter in txt
    for i in range(len(contours)): # for each contour region, save cropped img
        roi = getContour(nimg, contours[i], 0) # getting boxed roi (region of interest)
        lab = labels[ii]
        # imagename = linenum_wordnum_word_charnum_char.png
        cv2.imwrite('{}_{}_{}.png'.format(filename[:-4], i, lab), roi) # save cropped output image
        cv2.drawContours(img, contours, i, (0,0,255), thickness=thck) # draw boundary on full nonbinary img
        ii += 1
    if sanity == 0: print('Finished:', filename, '** NO SANITY CHECKS **')
    else: print('Finished:', filename)
    cv2.imwrite(filename[:-4]+'_segs.png', img) # save full image with regions drawn in red


### execute / testing ###

# your_path_here = '/Users/ovoowo/Desktop/fraktur'
# your_path_here = '/Users/Christine/cs/fraktur'
# seg_dir = your_path_here+'/segmentation/'

# images1 = ['a.png', 'b.png', 'hi.png'] # testing images
# images2 = ['hard.png', 'hard2.png', 'hoff.png']

# os.chdir(seg_dir+'/test_data') # location of img
# seg('hard2.png')
# seg('hi.png')
# for img in images2+images1:
#     os.chdir(seg_dir+'/test_data') # location of img
#     seg(img, 0, os.getcwd(), seg_dir+'/letters/testing')
