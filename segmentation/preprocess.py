# Christine Yang
# Fraktur Cracker
# preprocess.py
# preprocess an image, making it grayscale and binary

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os, numpy as np, cv2

# takes an image, returns original image, flattened image, and binary inverted image
def preprocess(filename):
    # import image #
    img = cv2.imread(filename)
    nimg = np.array([[x[0] for x in r] for r in img]) # flattened img matrix
    # preprocessing #
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    if len(np.unique(nimg)) > 2: # if img not binary
        _, nimg =  cv2.threshold(nimg, 150, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        nimg = cv2.morphologyEx(nimg, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('nimg.png', nimg)
    _, bin =  cv2.threshold(nimg, 150, 255, cv2.THRESH_BINARY_INV) # convert to binary inverted
    _, nimg =  cv2.threshold(nimg, 150, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('bin.png', bin)
    return (img, nimg, bin)

# os.chdir('/Users/ovoowo/Desktop/fraktur/segmentation/test_data/')
# os.chdir('/Users/Christine/cs/fraktur/segmentation/test_data/')
# images = ['a.png', 'b.png', 'hi.png', 'hard.png', 'hard2.png', 'hoff.png']
# filename ='hi.png'
# preprocess(filename)
