#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:56:51 2022

@author: max
"""

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

import os
# import json
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans

# %% 

with open('data/featuresnp.pickle', 'rb') as handle:
    featuresnp = pickle.load(handle)

with open('data/names.pickle', 'rb') as handle:
    names = pickle.load(handle)

with open('data/categories.pickle', 'rb') as handle:
    categories = pickle.load(handle)

with open('data/audios.pickle', 'rb') as handle:
    audios = pickle.load(handle)

with open('data/X_PCA.pickle', 'rb') as handle:
    X_PCA = pickle.load(handle)

with open('data/X_MDS.pickle', 'rb') as handle:
    X_MDS = pickle.load(handle)

with open('data/X_TSNE.pickle', 'rb') as handle:
    X_TSNE = pickle.load(handle)

# %% 

# X_embedded = X_PCA
# X_embedded = X_MDS
X_embedded = X_TSNE

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, n_init=30, max_iter=1000, verbose=1)
km_labels = kmeans.fit_predict(X_embedded)
km_centers = kmeans.cluster_centers_

# %% save clusters

with open('data/n_clusters.pickle', 'wb') as handle:
    pickle.dump(n_clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/km_labels.pickle', 'wb') as handle:
    pickle.dump(km_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/km_centers.pickle', 'wb') as handle:
    pickle.dump(km_centers, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% plot clusters
plt.clf()
plt.scatter( X_embedded[:,0], X_embedded[:,1], c=km_labels, alpha=0.5, s=3 )
for i,c in enumerate(km_centers):
    plt.text(c[0], c[1], 'c_'+str(i), color='red', alpha=0.5)
plt.savefig( 'data/tsne_clusters0.png', dpi=500 )