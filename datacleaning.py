# # # # # # # # # # # # # #
# Yuezhen Chen
# Please Work
import os,numpy as np,shutil
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
datapath = your_path_here+'fraktur/features/'
os.chdir(datapath)
from getFeatures import getFeats,txtGenerator
storepath = your_path_here +'fraktur/cleaning/' #the place to store imgs/error message
folder = 'data/letter_data/' #the one store imgs we want to clean
imgpath =your_path_here +'fraktur/'+folder #the place to get imgs'feature

from relabel import reLabel
from KCluster import elbow, kMeanclf
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/')

# elbow(X,storepath,storename)
