# Christine Yang
# Fraktur Cracker
# seg_the_third.py
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
from labelData import labelData
from wordSeg import wordSeg


your_path_here = '/Users/ovoowo/Desktop/fraktur'
# your_path_here = '/Users/Christine/cs/fraktur'
os.chdir(your_path_here+'/segmentation')
main_dir = os.getcwd()
np.set_printoptions(threshold=sys.maxsize) # print full np arrays, untruncated
divorced = ['u', 'ů', 'ü', 'ù', 'û', 'n', 'm', 'w'] # letters than tend get divorced
blenders = ['e', 'è', 'é', 'ê', 'ë', 'r', 'v', 'ſ', 't'] # letters that tend blend into next
images1 = ['a.png', 'b.png', 'hi.png'] # testing images
images2 = ['hard.png', 'hard2.png', 'hoff.png']

# get correct labels from .txt transcription
# where=0 for non word-segmented imgs, where=1 for word-segemented imgs
def getLabels(filename):
    try: # non word-segmented imgs
        int(filename[-5])
        txt_file = filename[:-4]+'.gt.txt' # get .txt filename
        txt_file = re.sub('\.nrm', '', txt_file)
        ein = open(txt_file, 'r') # open .txt file
        raw = ein.read().rstrip() # read .txt file
        ein.close()
        txt = re.sub('[.,\'\"“„ ]', '', raw) # replace spaces and punctuation
    except ValueError: # word-segmented imgs
        index = filename.rfind('_') # index of last occurrence of '_' in filename
        txt = filename[index+1:-4] # letter label of this img
        txt = re.sub(u'[.,\'\"“„-]', '', txt) # replace punctuation
    # add # chars for extra letters at the end from bad segmentation
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
        elif chars_str[i] == 'ͤ': # treat this as diacritic, not separate letter
            chars_ls = chars_ls[:-1]
            chars_ls.append(chars_str[i-1]+chars_str[i])
        else: chars_ls.append(chars_str[i])
        i += 1
    return chars_ls # list of letter labels


# get binary img matrix with just the contour filled in
# inv=0 for non-inverted, inv=1 for inverted
def getContour(nimg, contour, inv):
    if inv == 0:
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, [contour], 0, 0, -1) # color contour area as 0
        out = np.full_like(nimg, 255) # create array of 255s (whites)
        out[mask == 0] = nimg[mask == 0] # where mask is 0, change out value to img value
        x, y, w, h = cv2.boundingRect(contour) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting boxed roi (region of interest)
    else:
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, [contour], 0, 0, -1) # color contour area as 0
        out = np.full_like(nimg, 0) # create array of 0s (black)
        out[mask == 0] = 255  # where mask is 0, change out value to white
        x, y, w, h = cv2.boundingRect(contour) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting boxed roi (region of interest)
    return roi

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
        # index of first 255 pixel going down from xx,yy, aka stopping point
        try: stop = y+h + np.where(bin_orig[:,xx][y+h+1:]==255)[0][0]
        except: stop = len(bin_orig)
        # replace 0s in bin with 255s from contour center down to first 0
        bin[:,xx][yy:stop+1] = 255
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
    else: ratioThresh = 0.7 # set ratioThresh for u's and n's
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
    if lab in ['e', 'r', 'ſ', 't']: ratioThresh = 0.85
    if lab in ['v']: ratioThresh = 1
    ratio = w / h # width/height ratio of char image
    change = 0
    if lab == 'ſ':
        bottom_right = temp[h//2:][w//2:] # bottom right quarter of img
        top_right = temp[h//20:h//2][:,w//2:] # top right with top 5% cut off
        blacks = [np.count_nonzero(x) for x in bottom_right] # count black pixels per row
        if sum(blacks) > bottom_right.size*0.5: # if a good number of blacks in this area,
            change = 1
            blacks = np.array([np.count_nonzero(x) for x in top_right])
            blacks = np.where(blacks==0, 999, blacks)
            # bin index of row in this char with fewest black pixels
            boundary_i = np.ma.array(blacks).argmin() + y + h//20
            bin[boundary_i][x+w//2:x+w] = 0
    elif ratio > ratioThresh: # if above ratioThresh, split it up
        change = 1
        mid = temp[:,(w//10):(w-w//10)] # cut off the left and right 10% of the char img
        blacks = [np.count_nonzero(x) for x in mid.T] # count black pixels per col
        # bin index of col in this char with fewest black pixels
        boundary_i = np.array(blacks).argmin() + x + w//10
        bin[:,boundary_i] = 0
    return (bin, change)

# check for trash contours, return 1 if trash, 0 if not
def checkTrash(contours, nimg, bin):
    new_contours = []
    for iii in range(len(contours)): # remove trash contours
        x, y, w, h = cv2.boundingRect(contours[iii]) # get bounding box
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, contours, iii, 0, -1) # color contour area as 0
        # cv2.imwrite('what.png',bin[y:y+h, x:x+w])
        # if 40% of region is white, don't count it (prob inside of 'e', 'd', etc.)
        if np.count_nonzero(bin[mask == 0]==0) / len(bin[mask == 0]) > 0.6:
            continue # mostly white and also small
        # if it's really small, it's probably noise
        if h < len(nimg)/11: continue
        # if fairly small and in bottom half of image, probably noise/punctuation
        if h < len(nimg)/3:
            if y+h >= len(nimg)/2: continue
        new_contours.append(contours[iii]) # if passes all checks, add to new_contours
    return new_contours

# perform diacritic checks and sanity checks and morph img accordingly
# return new binary image
# sanity indicates whether or not to perform sanity checks
def morph(filename, destpath, sanity):
    labels = getLabels(filename) # get correct labels
    img, nimg, bin = preprocess(filename) # get img, flat img, and binary inverted img
    bin_orig = bin.copy()
    # find contour regions
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
        if sanity is 1: # if we wanna perform sanity checks
            ### SANITY CHECK for DIVORCEES ###
            if lab in divorced: # check for commonly divorced letters
                if divorce_flag != 1: # =1 means connection already drawn from other half of divorced chunk
                    bin, divorce_flag = sanityCheckDivorce(bin, bin_orig, dims, lab)
                else: divorce_flag = 0 # exit divorce_flag
            ### SANITY CHECK for BLENDING ###
            elif lab in blenders: # check for commonly blended letters
                bin, blender_flag = sanityCheckBlend(bin, lab, contours[i])
            # make sure labels matched correctly, otherwise don't do sanity checks
            if lab == '#':
                cv2.imwrite(destpath+'/###'+filename[:-4]+'_morphSANITY.png', bin)
                return morph(filename, destpath, 0)
            if i is len(contours)-1 and labels[ii+1] != '#':
                if blender_flag != 1:
                    cv2.imwrite(destpath+'/###'+filename[:-4]+'_morphSANITY.png', bin)
                    return morph(filename, destpath, 0)
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
    if sanity == 0: print('Finished:', filename, '** NO SANITY CHECKS **')
    else: print('Finished:', filename)
    return (img, nimg, bin, sanity)

# # save image for each segmented char, and full image with char boundaries drawn on
def seg(filename, datapath, destpath, thck=2):
    os.chdir(datapath)
    # orig img, flat img, and binary inverted img adjusted for diacritic and sanity checks
    img, nimg, bin, sanity = morph(filename, destpath, 1)
    labels = getLabels(filename) # get correct labels
    if sanity == 0: filename = '###' + filename
    os.chdir(destpath) # dir to save cropped letter images
    _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contour regions
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours = checkTrash(contours, nimg, bin)
    ii = 0 # index of letter in txt
    for i in range(len(contours)): # for each contour region, save cropped img
        # ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        # mask = cv2.drawContours(ones, contours, i, 0, -1) # color contour area as 0
        # out = np.full_like(nimg, 255) # create array of 255s (whites)
        # out[mask == 0] = nimg[mask == 0]  # where mask is 0, change out value to img value
        # x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box for cropping
        roi = getContour(nimg, contours[i], 0) # getting boxed roi (region of interest)
        lab = labels[ii]
        filename = re.sub('\.nrm', '', filename)
        cv2.imwrite('{}_{}_{}.png'.format(filename[:-4], i, lab), roi) # save cropped output image
        cv2.drawContours(img, contours, i, (0,0,255), thickness=thck) # draw boundary on full nonbinary img
        ii += 1
    cv2.imwrite(filename, img) # save full image with regions drawn in red

# execute
# os.chdir(main_dir+'/test_data') # location of img
# seg('hard2.png')
# seg('hi.png')
# for img in images2+images1:
#     os.chdir(main_dir+'/test_data') # location of img
#     seg(img)

moot = 0
os.chdir(your_path_here + '/data') # path to folder with book folders of data
# for each book folder in 'data'
for foldername in [x for x in os.listdir() if x[-3:] != 'txt' and x[0] != '.']:
    try: # make folder in 'letters' to store letter imgs for this book
        os.mkdir(main_dir+'/letters/'+foldername)
    except: pass
    datapath = your_path_here+'/data/'+foldername # path to img/txt data for this book
    destpath = main_dir+'/letters/'+foldername # path to save segmented letter imgs for this book
    os.chdir(datapath)
    for img in [x for x in os.listdir() if x[-3:] == 'png']: # for each line img in this book
        try: os.mkdir('temp') # to store word images
        except: pass
        worddatapath = os.getcwd()+'/temp'
        wordSeg(img, worddatapath) # get word segmented images
        for wordImg in os.listdir(): # for each word img for this line
            seg(wordImg, worddatapath, destpath) # segment letters
        shutil.rmtree(worddatapath)
        os.chdir(datapath)
        moot += 1
        if moot > 5: break
    labelData(destpath) # separate letter imgs into folders
    break

# os.system('git add .')
# os.system('git commit -m "ran seg on more data"')
# os.system('git push')
