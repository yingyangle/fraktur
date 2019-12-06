# Christine Yang, Yuezhen Chen
# Fraktur Cracker
# getFeatures.py
# get features for each char image

# https://pdfs.semanticscholar.org/6d50/6c0c85cda0ab43b47f997d8c179986e1ba5a.pdf

# features include:

# - zoning: split image into 16 sections
#       - black pixels in section / # total pixels in section
#       - black pixels in section / # total pixels in whole image
# - height/width ratio of image
# - distance profile features
#       - # pixels (distance) from upper edge of image to outer edge of char
#       - # pixels (distance) from lower edge of image to outer edge of char
#       - # pixels (distance) from left edge of image to outer edge of char
#       - # pixels (distance) from right edge of image to outer edge of char

import os, numpy as np, pickle, cv2
from zoning_YC import blackPerSect, blackPerImg, getDistance


# from distance import getDistance
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'



# get features for a char image


def getFeats(datapath,filename,n):
    your_path_here = '/Users/ovoowo/Desktop/fraktur/'
    os.chdir(your_path_here+'features/')
    customdict = pickle.load(open('dictionary.sav','rb'))
    os.chdir(datapath)

    img = cv2.imread(filename)
    num_rows, num_cols, _ = img.shape
    size = np.array([num_rows / num_cols]) # width/height ratio of image
#    blackS = blackPerSect(filename) # list of black ratios for each section
    black = blackPerImg(filename,n) # list of black ratios for each section over the whole image
    dist = getDistance(filename,n) # edge to char distance
    temp = filename[:-4]
    id = 0
    while id != -len(temp):
        id -= 1
        if temp[id] =='_':
            label = temp[id+1:]
            break
    ############finished#############
    if len(label) != 1:
        label = customdict[label]
    else:
        label =np.array([ord(label)])
    return (black,dist,label)

# execute
#datapath to a letter folder that store image
def txtGenerator(storepath,datapath,mode,foldername,n): #mode = 0 no label, mode = 1 with labels
    os.chdir(datapath)
    Bdataset = []
    Ddataset = []
    letters = []
    nImg = len(os.listdir())
    tracker = 0
    exceptions = [] #store error images

    #Error handler
    name = 'errors.txt'

    for filename in [x for x in os.listdir() if x[-3:] == 'png']:
        tracker += 1
        try:
#            letters.append(filename[-5:-4])
            (black,dist,label) = getFeats(datapath,filename,n)
            if mode == 1: # with label
                Bdata = np.concatenate((black,label))
                Ddata = np.concatenate((dist,label))
                Bdataset.append(Bdata)
                Ddataset.append(Ddata)
            else:
                Bdataset.append(black)
                Ddataset.append(dist)
        except Exception as e:
            print(e)
            exceptions.append(filename)
            aus = open(storepath+foldername+name, 'a')
            aus.write(filename + '\n')
            aus.close()
        if nImg < 500:
            if tracker % 100 == 99:
                print(str(tracker)+' images/ '+str(nImg)+' images done')
        else:
            if tracker % 500 == 499:
                print(str(tracker)+' images/ '+str(nImg)+' images done')
    # freq ={} #Get the frequency
    # for l in letters:
    #     keys = freq.keys()
    #     if l in keys:
    #         freq[l] += 1
    #     else:
    #         freq[l] = 1
    # print('Total number of chars = ',len(freq.keys()))
    # print ("Char - frequency : \n", freq)
    Btestdata = np.array(Bdataset)
    Dtestdata = np.array(Ddataset)
    print('Feature extraction for '+str(tracker)+' images done')
    print('='*40+'\n'+'Error images:\n')
    temp = [print(x) for x in exceptions]
    print('Total '+str(len(exceptions))+' Error \n'+'='*40)

    np.savetxt(storepath+foldername+str(n)+'_b.txt',Btestdata, delimiter=', ', fmt='%12.8f')
    np.savetxt(storepath+foldername+str(n)+'_d.txt',Dtestdata, delimiter=', ', fmt='%12.8f')
    return
