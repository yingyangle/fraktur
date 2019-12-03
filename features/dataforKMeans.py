import os,numpy as np
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
modulepath = your_path_here +'fraktur/features/'
os.chdir(modulepath)
from getFeatures import getFeats,txtGenerator

path =your_path_here + 'fraktur/data/letter_data/'
os.chdir(path)

subFolders= os.listdir(path)
Folder = subFolders[1]
print(Folder)
datapath = path+Folder
txtGenerator(datapath,0,Folder,8)
