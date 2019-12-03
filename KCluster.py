from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
# https://towardsdatascience.com/clustering-based-unsupervised-learning-8d705298ae51
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/')

txts = [x for x in os.listdir() if x[-8:] == 'data.txt']

filename = txts[0]
X = np.loadtxt(filename, delimiter = ',')
(numSamples, numFeatures) = X.shape
print((numSamples, numFeatures))
'''
wcss = []
K = list(range(1,7))
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X)
#    labels = kmeans.predict(X)
    wcss.append(kmeans.inertia_)
plt.plot(K, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.savefig('elbow.png',dpi=200)
#print(labels)

Find the elbow point according to the graph and then use it as n_clusters
'''


# Fitting K-Means to the dataset
#KMeans
km = KMeans(n_clusters=3)
km.fit(X)
km.predict(X)
labels = km.labels_

#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("0 section blackness per ")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)
