# -*- coding: utf-8 -*-
# Christine Yang
# Fraktur Cracker
# preprocess.py
# preprocess an image, making it grayscale and binary

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os, numpy as np, cv2, sys


# takes an image, returns original image, flattened image, and binary inverted image
def preprocess(filename):
    # import image #
    img = cv2.imread(filename)
    nimg = np.array([[x[0] for x in r] for r in img]) # flattened img matrix
    # preprocessing #
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    if len(np.unique(nimg)) > 2: # if img not binary
        _, nimg =  cv2.threshold(nimg, 150, 255, cv2.THRESH_BINARY)
        kernel1 = np.ones((4,3), np.uint8)
        kernel2 = np.ones((3,2), np.uint8)
        erosion = cv2.erode(nimg, kernel1, iterations = 1) # erode img
        # cv2.imwrite('erode.png', erosion)
        dilation = cv2.dilate(erosion, kernel2, iterations = 1) # dilate img
        # cv2.imwrite('dilate.png', dilation)
        nimg = dilation
    _, bin =  cv2.threshold(nimg, 150, 255, cv2.THRESH_BINARY_INV) # convert to binary inverted
    # cv2.imwrite('nimg.png', nimg)
    return (img, nimg, bin)


### execute / testing ###

# os.chdir('/Users/ovoowo/Desktop/fraktur/segmentation/test_data')
# os.chdir('/Users/Christine/cs/fraktur/segmentation/test_data')
# test_dir = os.getcwd()
# np.set_printoptions(threshold=sys.maxsize) # print full np arrays untruncated, for testing
# images = ['a.png', 'b.png', 'hi.png', 'hard.png', 'hard2.png', 'hoff.png']
# filename ='rrr.png'
# img, nimg, bin = preprocess(filename)
