import os, numpy as np, shutil
from os.path import join

# your_path_here = '/Users/ovoowo/Desktop/fraktur'
your_path_here = '/Users/Christine/cs/fraktur'

datapath = join(your_path_here, 'data/3books/')
despath = join(your_path_here, 'data/3books_letters')

subFolders= os.listdir(datapath)

# for each letter folder
for folder in [x for x in subFolders if x[0] != '.']: 
    os.chdir(join(datapath, folder)) # access every folder
    # get all the good images without "No Sanity Check"
    images = np.array([x for x in os.listdir() if x[0:3] != '###']) 
    for img in images: # for each image
        os.rename(img, join(despath, img)) # move image to new folder
