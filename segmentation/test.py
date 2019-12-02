# Christine Yang
# Fraktur Cracker
# test.py
# random testing file

import cv2, os, shutil, numpy as np
from seg import checkTrash, getContour, seg, fillInSpaces
from preprocess import preprocess

os.chdir('/Users/Christine/Documents/cs/fraktur/segmentation/test_data')

try: shutil.rmtree('test')
except FileNotFoundError: pass
os.mkdir('test')


### testing contours / trash ###

def test1(filename):
    img, nimg, bin = preprocess(filename)
    _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort contours in order
    bin = fillInSpaces(bin, contours)

    contours = checkTrash(contours, nimg, bin)

    os.chdir('test')
    i = 0
    for c in contours:
        img = getContour(nimg, c, 0)
        cv2.imwrite(str(i)+filename, img)
        i += 1


### testing seg.py ###

def test2(filename):
    seg(filename, 0, os.getcwd(), os.getcwd()+'/test')


### execute ###

filename = 'white.png'
filename = 'rrr.png'
filename = 'uu.png'
filename = 'uuu_ſeiner.png'
filename = 'ess.png'
filename = 'ess1_Gelüſte.png'

# test1(filename)
seg(filename, 1, os.getcwd(), os.getcwd()+'/test')



# img, nimg, bin = preprocess(filename)
#
# _, contours, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort contours in order
# contour = contours[0]
# cv2.imwrite('bin.png', bin)
#
# out = np.full_like(bin, 0) # create array of 0s (blacks)
# for c in contours:
#     contour = c
#     ones = np.ones_like(bin) # create array of 1s (almost blacks)
#     mask = cv2.drawContours(ones, [contour], 0, 0, -1) # color contour area as 0
#     out[mask == 0] = 255 # where mask is 0, change out value to img value
#     x, y, w, h = cv2.boundingRect(contour) # get bounding box for cropping
#     roi = out[y:y+h, x:x+w] # getting boxed roi (region of interest)
# cv2.imwrite('help.png', out)
