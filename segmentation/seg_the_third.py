# Christine Yang
# Fraktur Cracker
# seg_the_third.py
# third attempt at character segmentation, using OpenCV

# OPENCV
# https://stackoverflow.com/questions/49291770/ocr-cropping-the-letters
# https://stackoverflow.com/questions/42316315/how-to-segment-license-plate-characters-removing-unwanted-characters-using-openc
# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import os, codecs, numpy as np, cv2

os.chdir('/Users/Christine/cs/fraktur/segmentation')
main_dir = os.getcwd()

# save each segmented character as separate image, and full image with boundaries drawn on
def seg(filename):
    img = cv2.imread(filename) # import image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # change to grayscale
    os.chdir('letters') # dir to save cropped letter images
    ret, thresh = cv2.threshold(gray_img, 127, 255, 1)
    _, contours , _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contour regions
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i in range(len(contours)): # for each contour region, save cropped image
        ones = np.ones_like(img) # create array of 1s (almost blacks)
        mask = cv2.drawContours(ones, contours, i, 0, -1) # color contour area as 0
        if np.count_nonzero(img[mask == 0]==255)/len(img[mask == 0]) > 0.8:
            continue # if 90% of region is white
        if len(img[mask == 0]) < 1000: continue # if region really small, it's prob noise
        out = np.full_like(img, 255) # create array of 255s (whites)
        out[mask == 0] = img[mask == 0] # where mask is 0, change out value to img value
        x, y, w, h = cv2.boundingRect(contours[i]) # get bounding box for cropping
        roi = out[y:y+h, x:x+w] # getting ROI
        cv2.imwrite('{}{}.png'.format(filename[:-4], i), roi) # save cropped output image
        cv2.drawContours(img, contours, i, (0,0,255), thickness=2)
    cv2.imwrite(filename, img) # full image with regions drawn in red

# execute
seg('hard.png')
