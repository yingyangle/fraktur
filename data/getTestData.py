import os,numpy as np
import shutil

your_path_here = '/Users/ovoowo/Desktop/fraktur'
#your_path_here = '/Users/Christine/cs/fraktur'
path =your_path_here+'/data/letter_data/'
datapath =your_path_here+'/data/dataset'

subFolders= os.listdir(path)

counter = 0
totalNum = 0
n = len(subFolders)-1
folderTrack = 0
for folder in subFolders:
    folderTrack += 1
    if folder != '.DS_Store':
        os.chdir(path+folder) #access every folder
        imgs = np.array([x for x in os.listdir() if x[0:3] != '###']) #get all the good images without "No Sanity Check"
        total = len(imgs)
        for img in imgs:
            shutil.copy(img, datapath) #copy every image to the dataset folder
            counter += 1
            print(str(counter)+' images/ '+str(total)+' images moved\t' + 'In '+ str(folderTrack)+'th folder out of '+str(n)+' folder')
        totalNum += counter
        counter = 0
print(str(totalNum)+' images move')
