import cv2, os, numpy as np

for filename in [x for x in os.listdir() if x[-3:] == 'png' and x[0] != '.']:
    img = cv2.imread(filename)
    print(filename)
    print(img.size, np.count_nonzero(img), np.count_nonzero(img)/img.size)