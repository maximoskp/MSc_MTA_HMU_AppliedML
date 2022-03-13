#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:08:28 2022

@author: max
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import dimensionality_reduction_methods as drm

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/featuresnp.pickle', 'rb') as handle:
    featuresnp = pickle.load(handle)

with open('data/domiantQsnp.pickle', 'rb') as handle:
    domiantQsnp = pickle.load(handle)

with open('data/happyflagsnp.pickle', 'rb') as handle:
    happyflagsnp = pickle.load(handle)

with open('data/energeticflagsnp.pickle', 'rb') as handle:
    energeticflagsnp = pickle.load(handle)

X_embedded = drm.reduce_dimensions(featuresnp[:,:-1], method='TSNE')

# %% 

plt.clf()
plt.scatter( X_embedded[:,0], X_embedded[:,1], c=domiantQsnp, cmap='jet', alpha=0.5 )
# plt.plot( X_embedded[:,0], X_embedded[:,1], '.' )
# for i,lab in enumerate(domiantQsnp):
#     plt.text( X_embedded[i,0], X_embedded[i,1], str(lab) )

# %% 

plt.clf()
plt.scatter( X_embedded[:,0], X_embedded[:,1], c=happyflagsnp, cmap='PiYG', alpha=0.5 )
plt.title('Happy')

# %% 

plt.clf()
plt.scatter( X_embedded[:,0], X_embedded[:,1], c=energeticflagsnp, cmap='PiYG', alpha=0.5)
plt.title('Energetic')
