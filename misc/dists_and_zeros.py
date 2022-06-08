# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:03:50 2022

@author: user
"""

import numpy as np
import scipy.spatial as scp
import matplotlib.pyplot as plt

x = np.array([[1,1,0,0],
              [0,1,2,0],
              [0,0,1,2],
              [0,0,2,1]])
'''
x = np.array([[1,10,0,0],
              [10,1,0,0],
              [0,1,10,0],
              [0,1,1,0],
              [0,0,1,10],
              [0,0,1,1]])
'''
d = scp.distance_matrix( x , x )


# %% 

from sklearn.decomposition import NMF

model = NMF(n_components=2, init='random', random_state=0, max_iter=2000)

W = model.fit_transform(x.T)

H = model.components_.T

xx = np.matmul(W,H.T).T

# %% 

plt.clf()
for i in range(H.shape[0]):
    plt.plot(H[i,0], H[i,1], 'x')
    plt.text(H[i,0], H[i,1], str(i))
plt.show()

# %% 
'''
H04 = np.linalg.norm(H[:,0] - H[:,4] )
H05 = np.linalg.norm(H[:,0] - H[:,5] )

'''
# %% 

from sklearn.manifold import MDS

embedding = MDS(n_components=2)
M = embedding.fit_transform(x)
'''
M04 = np.linalg.norm(M[0,:] - M[4,:] )
M05 = np.linalg.norm(M[0,:] - M[5,:] )
'''

# %%

plt.clf()
for i in range(M.shape[0]):
    plt.plot(M[i,0], M[i,1], 'x')
    plt.text(M[i,0], M[i,1], str(i))
plt.show()