import os
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # Labeling the images with new Label  # # # # # #
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
path =your_path_here + 'data/letter_data/'
# os.chdir(path)
# subFolders= os.listdir(path)
# Folder = subFolders[1]
# datapath = path+Folder
# img = [x for x in os.listdir(datapath) if x[-3:] == 'png']
#os.chdir(datapath)

def reLabel(errorpath,labelsB,labelsD,foldername,datapath):
    if errorpath != 'noerror':
        os.chdir(errorpath)
        errorHandler = foldername+'errors.txt'
        file = open(errorHandler, 'r') # open .txt file
        raw = file.readlines() # read .txt file return lists of error image with \n
        errors = [raw[i][:-1] for i in range(len(raw))]
        for error in errors:
            os.remove(datapath+error)
    os.chdir(datapath)
    img = [x for x in os.listdir(datapath) if x[-3:] == 'png']
    print(len(img))
    bound = len(img) - 1
    for i in range(0,bound):
        os.rename(img[i],img[i][:-4]+str(labelsB[i])+str(labelsD[i])+img[i][-4:])
    return
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# datapath =your_path_here+'data/letter_data/100d'
# img = [x for x in os.listdir(datapath) if x[-3:] == 'png']
# os.chdir(your_path_here)
# errorHandler = '100d'+'errors.txt'
# file = open(errorHandler, 'r') # open .txt file
# raw = file.readlines() # read .txt file
# raw[0][:-1]
# error = [raw[i][:-1] for i in range(len(raw))]
