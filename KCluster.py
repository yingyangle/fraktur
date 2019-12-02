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


your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
os.chdir(your_path_here+'fraktur/letters_for_testing/')


filename = 'testdata.txt'
X = np.loadtxt(filename, delimiter = ',')
(numSamples, numFeatures) = X.shape
print((numSamples, numFeatures))
print(X)

K = 30
kmeans = KMeans(n_clusters=K).fit(X)
labels = kmeans.predict(X)
print(labels)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 1, pred[:, 0].max() + 1
y_min, y_max = pred[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(X)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# data = dataset[:,range(2)].reshape((numSamples, 2))
# print(data)
# K = range(1,52)
#
# GMM = [GaussianMixture(n_components=k).fit(X) for k in K]
# print("Trained EM models")
# LL = [gmm.score(X) for gmm in GMM]
# print("Calculated the log likelihood for each k")
# BIC = [gmm.bic(X) for gmm in GMM]
# print("Calculated the BICs for each K")
# AIC = [gmm.aic(X) for gmm in GMM]
# print("Calculated the AICs for each K")
# min = np.amin(np.array(BIC))
# result = np.where(np.array(BIC) == min)
# print('the minimum BIC = '+str(min))
# plt.style.use('ggplot')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(K, BIC, 'c*-', label='BIC')
# ax.plot(K, AIC, 'b*-', label='AIC')
# plt.grid(True)
# plt.xlabel('Number of clusters')
# plt.ylabel('Inference scores')
# plt.legend(loc='best')
# plt.title('Bayesian and Akaike Information Criterion Curve')
# plt.show()
# fig = plt.figure()
# plt.show()
