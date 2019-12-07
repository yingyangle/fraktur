import os, shutil, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from relabel import reLabel

your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
os.chdir(your_path_here)
datapath = your_path_here+'features/'

def fetchFeature(filename,path):
    os.chdir(path)
    X = np.loadtxt(filename, delimiter = ',')
    return X
#(numSamples, numFeatures) = X.shape
#print((numSamples, numFeatures))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Elbow method:Decide n_cluster  # # # # # # #
# def elbow(X,storepath,storename):
#     wcss = []
#     K = list(range(1,7))
#     for k in K:
#         kmeans = KMeans(n_clusters=k).fit(X)
#     #    labels = kmeans.predict(X)
#         wcss.append(kmeans.inertia_)
#     fig = plt.figure()
#     plt.plot(K, wcss)
#     plt.title('The Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('WCSS')
#     plt.savefig(storename,dpi=200)
#     return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # silhouette: Decide n_cluster  # # # # # # #
# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
def silhouette(x):
    sil = []
    kl =list(range(2,7)) #change upper bound to test different thing
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in kl:
        kmeans = KMeans(n_clusters = k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    ind = np.argmax(np.array(sil))
    numCluster = np.array(kl[ind])
    return numCluster

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # Plot # # # # # # # # # # # # # #
def plot(X):
    fig = plt.figure(1, figsize=(10,10))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
              c=labels.astype(np.float), edgecolor="k", s=50)
    ax.set_xlabel('0 section blackness per image')
    ax.set_ylabel('1 section blackness per image')
    ax.set_zlabel('2 section blackness per image')
    plt.title("K Means", fontsize=14)
    plt.savefig('KMeans.png',dpi=200)
    return
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Fitting K-Means to the dataset

def kMeanclf(X,n):
    km = KMeans(n_clusters=n)
    km.fit(X)
    km.predict(X)
    labels = km.labels_
#    plot(X)
    counts = np.bincount(labels)
    idLabel = np.argmax(counts)
    return (idLabel,labels)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
