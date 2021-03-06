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

import os, codecs, numpy as np, cv2, re

os.chdir('/Users/Christine/cs/fraktur/segmentation')
main_dir = os.getcwd()
images1 = ['a.png', 'b.png', 'hi.png']
images2 = ['hard.png', 'hard2.png', 'hoff.png']

# get correct labels from .txt transcription
def getLabels(filename):
    os.chdir(main_dir)
    txt_file = filename[:-4]+'.txt' # get .txt filename
    ein = open('test_data/'+txt_file, 'r') # open .txt file
    raw = ein.read().rstrip() # read .txt file
    ein.close()
    txt = re.sub('[.,\'\" ]', '', raw) # replace spaces and punctuation
    # add # chars for extra letters at the end from bad segmentation
    return txt+'{0:#^50}'.format('')

filename = 'hoff.png'
# save each segmented character as separate image, and full image with boundaries drawn on
def seg(filename, thck=2):
    os.chdir(main_dir)
    img = cv2.imread('test_data/'+filename) # import image
    labels = getLabels(filename) # get correct labels
    os.chdir('letters') # dir to save cropped letter images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
    ret, thresh = cv2.threshold(gray_img, 127, 255, 1) # binary
    # thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imwrite(filename[:-4]+'_thresh.png', thresh)
    kernel = np.ones((1,1), np.uint8) # image dilation
    dil = cv2.dilate(thresh, kernel, iterations=1) # image dilation
    _, contours, _ = cv2.findContours(dil.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contour regions
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    nimg = np.array([[x[0] for x in r] for r in img]) # flatten img matrix
    ii = 0
    for i in range(len(contours)): # for each contour region, save cropped image
        ones = np.ones_like(nimg) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, contours, i, 0, -1) # color contour area as 0
        out = np.full_like(nimg, 255) # create array of 255s (whites)
        out[mask == 0] = nimg[mask == 0] # where mask is 0, change out value to img value
        x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting ROI
        if np.count_nonzero(dil[mask == 0]==0)/len(nimg[mask == 0]) > 0.75:
            continue # if 80% of region is white (0 = black, since dil is inverted)
        if len(roi) < len(nimg)/3: continue # if region really small, it's prob noise
        cv2.imwrite('{}_{}_{}.png'.format(filename[:-4], i, labels[ii]), roi) # save cropped output image
        cv2.drawContours(img, contours, i, (0,0,255), thickness=thck) # draw boundary on full img
        ii += 1
    cv2.imwrite(filename, img) # save full image with regions drawn in red
    os.chdir(main_dir)


# execute
# for img in images1: seg(img, 1)
for img in images2: seg(img, 2)
