# Christine Yang
# Fraktur Cracker
# preprocess.py
# preprocess an image, making it grayscale and binary

import os, numpy as np, cv2


# takes an image, returns original image, flattened image, and binary grayscale image
def preprocess(filename):
    img = cv2.imread(filename) # import image
    nimg = np.array([[x[0] for x in r] for r in img]) # flatten img matrix
    # preprocess image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY) # convert to binary
    bin = thresh
    cv2.imwrite('img.png', img)
    cv2.imwrite('nimg.png', nimg)
    cv2.imwrite('bin.png', bin)
    return (img, nimg, bin)
