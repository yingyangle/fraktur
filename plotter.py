import os, math, pickle, numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from KNN import knnClf, getData
from getFeatures import getFeats,txtGenerator

your_path_here = '/Users/ovoowo/Desktop/fraktur/'
# your_path_here = '/Users/Christine/cs/fraktur/'
datapath = your_path_here + 'features/'
txtpath = your_path_here + 'datatxt/'
os.chdir(datapath)

plotpath = your_path_here+'plot/'

# Plot k vs accuracy for the same N
# want to plot subplot to compare features:blackness and distance
def kAccuracy(ks,bAccuracies, dAccuracies,bPins,dPins,n):
    fig = plt.figure()
    plt.plot(np.array(ks), np.array(bAccuracies), 'p-', label = 'Blackness for Pout')
    plt.plot(np.array(ks), np.array(dAccuracies), 'cp-',label = 'Distance for Pout')
    plt.plot(np.array(ks), np.array(bPins), 'p:', label = 'Blackness for Pin')
    plt.plot(np.array(ks), np.array(dPins), 'cp:',label = 'Distance for Pin')
    plt.legend(loc='best')
    x = tuple(np.array(ks))
    by = tuple(np.around(np.array(bAccuracies),3))
    dy = tuple(np.around(np.array(dAccuracies),3))
    for xy in zip(x, by):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',fontsize=4)
    for xy in zip(x, dy):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data',fontsize=4)
    plt.xlabel('K')
    plt.ylabel('Precision')
    s ='Accuracy vs. K for N = '+str(n)
    plt.title(s)
    plt.savefig(plotpath +'PinvsPout'+str(n)+'.png',dpi=200)
    return

# plot N vs best k
def nk(bks, dks, n):
    fig = plt.figure()
    plt.plot(np.array(n), np.array(bks), 'p-', label = 'Blackness')
    plt.plot(np.array(n), np.array(dks), 'cp-',label = 'Distance')
    plt.legend(loc='best')
    x = tuple(np.array(n))
    by = tuple(np.array(bks))
    dy = tuple(np.array(dks))
    for xy in zip(x, by):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    for xy in zip(x, dy):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.xlabel('N for feature dimension')
    plt.ylabel('best K with highest In sample accuracy')
    s ='N vs best k'
    plt.title(s)
    plt.savefig(plotpath +'nk.png',dpi=200)
    return

# plot N vs best Accuracy for two features
def nbestkAccuracy(n,bestbAccuracy,bestdAccuracy,bestbKs,bestdKs):
    fig = plt.figure()
    plt.plot(np.array(n), np.array(bestbAccuracy), 'p-', label = 'Blackness')
    plt.plot(np.array(n), np.array(bestdAccuracy), 'cp-',label = 'Distance')
    plt.legend(loc='best')
    x = tuple(np.array(n))
    by = tuple(np.around(np.array(bestbAccuracy),3))
    dy = tuple(np.around(np.array(bestdAccuracy),3))
    blabels = tuple(np.array(bestbKs))
    dlabels = tuple(np.array(bestdKs))
    for xy in zip(x, by):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    for xy in zip(x, dy):
        plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    ############# Label K bestbKs,bestdKs
    for x,by,dy, bk,dk in zip(x,by,dy, blabels,dlabels):
        blabel = 'K = %d'% bk
        plt.annotate(blabel, # this is the text
                     (x,by), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center')
        dlabel = 'K = %d'% dk
        plt.annotate(dlabel, # this is the text
                     (x,dy), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center')
    #############
    plt.xlabel('N for feature dimension')
    plt.ylabel('accuracy with best k')
    s =' N vs best Accuracy'
    plt.title(s)
    plt.savefig(plotpath +'nbestk.png',dpi=200)
    return

######################################################
########### Data Generator for different n ###########
# Folder = 'dataset'
# your_path_here = '/Users/ovoowo/Desktop/'
# #your_path_here = '/Users/Christine/cs/'
# os.chdir(your_path_here+'fraktur/')
# datapath = your_path_here+'fraktur/data/dataset'
# files = [] #os.lsdir(your_path_here+'fraktur/')
# for n in range(2,9):
#     txtGenerator(datapath,1,Folder,n)
#     Bfilename = Folder+str(n)+'_blacktestdata.txt'
#     files.append(Bfilename)
# temp =[print(file) for file in files]
######################################################
ns = list(range(4,9))
bestbAccuracy=[]
bestdAccuracy=[]
btxts = [x for x in os.listdir(txtpath) if x[-17:-12] == 'black']
dtxts = [x for x in os.listdir(txtpath) if x[-17:-12] == 'tance']
bestbKs = []
bestdKs = []
for n in range(len(btxts)):
    print('This is the training for '+str(ns[n])+' by '+str(ns[n])+' dimension feature:')
    bFile = btxts[n] # <- due to the nature of how txt in lsdir() arrange
    dFile = dtxts[n] # <- in alphabetical order, we know i is align with n
    (bx,by)=getData(bFile,txtpath)
    (dx,dy)=getData(dFile,txtpath)
    kbound = 10
    (bAccuracies,bPins) = knnClf(bx, by, kbound) #model, bestk, best accuracy, accuracies
    (dAccuracies,dPins) = knnClf(dx, dy, kbound)
    kVals = range(1,kbound,2)
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # plot k vs Accuracy with same n # # # #
    kAccuracy(kVals,bAccuracies, dAccuracies,bPins,dPins,ns[n])
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # plot n vs bestK # # # # # # # # plot outside for loop
    bestbK = kVals[np.argmax(bAccuracies)]
    bestdK = kVals[np.argmax(dAccuracies)]
    bestbKs.append(bestbK)
    bestdKs.append(bestdK)
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # nbestkAccuracy # # # # # # # # plot outside for loop
    bestbAccuracy.append(bAccuracies[np.argmax(bAccuracies)])
    bestdAccuracy.append(dAccuracies[np.argmax(dAccuracies)])
    # # # # # # # # # # # # # # # # # # # # # # # #


############################################################
# nk(bestbKs, bestdKs, ns)
# nbestkAccuracy(ns,bestbAccuracy,bestdAccuracy,bestbKs,bestdKs)
