from sklearn.cluster import KMeans
from sklearn import metrics
import os,shutil
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
datapath = your_path_here+'features/'
#from lettersCounter import letterCounter
from relabel import reLabel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/')

txts = [x for x in os.listdir() if x[-8:] == 'data.txt' and x[:4] != 'data']
filenameb = txts[0]
filenamed = txts[1]
Xb = np.loadtxt(filenameb, delimiter = ',')
Xd = np.loadtxt(filenamed, delimiter = ',')
#(numSamples, numFeatures) = X.shape
#print((numSamples, numFeatures))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Elbow method:Decide n_cluster  # # # # # # #
# def elbow(X):
#     wcss = []
#     K = list(range(1,7))
#     for k in K:
#         kmeans = KMeans(n_clusters=k).fit(X)
#     #    labels = kmeans.predict(X)
#         wcss.append(kmeans.inertia_)
#     plt.plot(K, wcss)
#     plt.title('The Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('WCSS')
#     plt.show()
#     plt.savefig('elbow.png',dpi=200)
#     #print(labels)
#     return
# elbow(X)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Fitting K-Means to the dataset
correctL = []
allL = []
def kMeanclf(X,n):
    km = KMeans(n_clusters=n)
    km.fit(X)
    km.predict(X)
    labels = km.labels_

    #Plotting
    fig = plt.figure(1, figsize=(10,10))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
              c=labels.astype(np.float), edgecolor="k", s=50)
    ax.set_xlabel('0 section blackness per image')
    ax.set_ylabel('1 section blackness per image')
    ax.set_zlabel('2 section blackness per image')
    plt.title("K Means", fontsize=14)
    plt.savefig('KMeans.png',dpi=200)

    counts = np.bincount(labels)
    return (np.argmax(counts),counts,labels)
(xbL, xbA,labelsB) = kMeanclf(Xb,3)
(xdL, xdA,labelsD) = kMeanclf(Xd,3)
correctL.append(xbL)
correctL.append(xdL)
allL.append(xbA)
allL.append(xdA)
print('For blackness, the largest cluster has label: ', correctL[0])
print('For blackness, all labels have: ', allL[0])
print('For distance, the largest cluster has label: ', correctL[1])
print('For distance, all labels have: ', allL[1])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # Labeling the images with new Label  # # # # # #
path =your_path_here+'fraktur/data/letter_data/'
os.chdir(path)
subFolders= os.listdir(path)
Folder = subFolders[1]
datapath = path+Folder
reLabel(labelsB,labelsD,Folder,datapath)
sourcepath = path+Folder
destpath =your_path_here+'fraktur/data/goodd'
os.chdir(sourcepath)#access folder of letter
imgs = np.array([x for x in os.listdir(sourcepath) if x[-6:] == '{}{}.png'.format(correctL[0],correctL[1]) and x[:3] !='###'])

###2nd try:
# allL[1] = np.array(allL[1].tolist().remove(correctL[1]))
# new = max(allL[1].tolist())

# imgs = np.array([x for x in os.listdir(sourcepath) if x[-6:] == '{}{}.png'.format(correctL[0],new) and x[:3] !='###'])
#imgs2 = np.array([x for x in os.listdir(sourcepath) if x[-6:] == '{}{}.png'.format(1,2) and x[:3] !='###'])
#imgs = np.concatenate(imgs,imgs2)
total = imgs.size
counter = 0
for img in imgs:
    shutil.copy(img, destpath) #copy every image to the dataset folder
    counter += 1
    print(str(counter)+' images/ '+str(total)+' images moved\t')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
