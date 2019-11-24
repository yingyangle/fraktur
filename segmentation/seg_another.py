# Christine Yang
# Fraktur Cracker
# seg_another.py
# attempt at character segmentation and also some playing around with Tesseract

# USING TESSERACT
# https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d
# https://www.quora.com/Is-it-possible-to-output-the-character-word-line-segmentation-using-Tesseract-OCR

# DOWNLOADING TESSERACT
# https://github.com/sirfz/tesserocr/issues/177
# https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
# ^^^ download fraktur file from here and copy to tessdata folder

# OPENCV
# https://stackoverflow.com/questions/49291770/ocr-cropping-the-letters
# https://stackoverflow.com/questions/42316315/how-to-segment-license-plate-characters-removing-unwanted-characters-using-openc
# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import os, codecs, numpy as np, tesserocr, cv2
from tesserocr import PyTessBaseAPI

os.chdir('/Users/Christine/cs/fraktur/segmentation')
main_dir = os.getcwd()

images = ['hard.png', 'hard2.png', 'a.png', 'b.png', 'hi.png', 'hoff.png']

# print Tesseract's transcription of a line of Fraktur
def tess(filename):
    # frk = german fraktur, psm=7 treats image as line of text
    with PyTessBaseAPI(lang='frk', psm=7) as api:
        # print(api.GetAvailableLanguages())
        # print(tesserocr.tesseract_version())
        api.SetImageFile(filename)
        with codecs.open(filename[:-3]+'txt', 'r', 'utf-8') as ans:
            print('Actual:', ans.read().rstrip()) # print correct label
        print('Prediction:', api.GetUTF8Text().rstrip()) # print Tesseract's guess
        print('Confidence:', str(api.AllWordConfidences()), '\n') # 0 worst, 100 best

# segment letters using OpenCV contours, save segmentations as image
def cv(filename):
    image = cv2.imread(filename) # import image
    try: # folder for cropped letter images
        os.mkdir('letters')
        os.chdir('letters')
    except: os.chdir('letters')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
    cv2.imshow('gray', gray)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV) # binary
    cv2.imshow('second', thresh)
    kernel = np.ones((1,1), np.uint8) # dilation
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated', img_dilation)
    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort contours
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr) # get bounding box
        roi = image[y:y+h, x:x+w] # getting ROI
        cv2.imshow('segment no:'+str(i),roi) # show ROI
        cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)
        if w > 15 and h > 15: # save contour as image
            cv2.imwrite('{}_roi{}.png'.format(filename[:-4], i), roi)
    cv2.imshow('marked areas', image) # show full image with contour boundaries
    cv2.imwrite(filename, image) # save full image with contour boundaries
    cv2.waitKey(0)
    os.chdir(main_dir)

# get Tesseract transcriptions
for img in images: tess(img)

# segment letters
seg('a.png')
