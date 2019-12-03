import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, math, pickle
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
datapath = your_path_here+'features/'
os.chdir(datapath)
from KNN import knnClf, getData
from getFeatures import getFeats,txtGenerator

def kAccuracy(ks,accuracies):
    fig = plt.figure()
    plt.plot(np.array(ks), np.array(accuracies), 'c.-')
    x = tuple(np.array(ks))
    y = tuple(np.array(accuracies))
    for xy in zip(x, y):
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('Average Precision')
    s ='Average Precision vs. K for best N = '+str(n)
    plt.title(s)
    plt.savefig('knnAccuracy'+str(n)+'.png',dpi=200)
    return

def zoningAccuracy(n,bestAccuracy,k):
    fig = plt.figure()
    plt.plot(np.array(n), np.array(bestAccuracy), 'p.-')
    x = tuple(np.array(n))
    y = tuple(np.array(bestAccuracy))
    for xy in zip(x, y):
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    fig.show()
    plt.xlabel('N for Feature Dimension(NxN)')
    plt.ylabel('Average Precision')
    s ='Average Precision vs. N'+str(k)
    plt.title(s)
    plt.savefig('nAccuracy'+str(k)+'.png',dpi=200)
    return


######################################################
########### Data Generator for different n ###########
Folder = 'dataset'
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/')
datapath = your_path_here+'fraktur/data/dataset'
files = [] #os.lsdir(your_path_here+'fraktur/')
for n in range(2,9):
    txtGenerator(datapath,1,Folder,n)
    Bfilename = Folder+str(n)+'_blacktestdata.txt'
    files.append(Bfilename)
temp =[print(file) for file in files]
######################################################

'''
ns = list(range(2,9))
bestAccuracy,k = []
for i in range(len(files)):
    file = files[i]
    (x,y)=getData(file)
    (model,bestK,score) = knnClf(x, y)
'''
############################################################
