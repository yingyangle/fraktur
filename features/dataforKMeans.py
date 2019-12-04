import os,numpy as np
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
modulepath = your_path_here +'fraktur/features/'
os.chdir(modulepath)
from getFeatures import getFeats,txtGenerator
storepath = your_path_here +'fraktur/cleaning/' #the place to store imgs/error message
folder = 'data/letter_data/' #the one store imgs we want to clean
imgpath =your_path_here +'fraktur/'+folder #the place to get imgs'feature
txtpath = storepath + 'letterfeatures/'
#get all the folder's name that need to be cleaned
os.chdir(imgpath)
subFolders= os.listdir(imgpath)[1:] #get rid of .DS_Store document
count = 0
total = len(subFolders)
for folder in subFolders:
    #os.chdir(imgpath+folder)
    txtGenerator(txtpath,imgpath+folder,0,folder,8) #get feature for every letter folder
    count += 1
    print('~'*20)
    print('Folder '+folder+' is done. \t '+str(count)+' out of '+str(total)+' folder left')
    print('~'*20)
