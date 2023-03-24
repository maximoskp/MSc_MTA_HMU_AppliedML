#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 08:01:02 2022

@author: max
"""

import numpy as np
import os

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

# %% 

with open('data/audio_structs.pickle', 'rb') as handle:
    audio_structs = pickle.load(handle)

# %% padding function

def pad_end_cut(array, s):
    """
    :param array: numpy array as a single row
    :param s: desired size
    :return: padded or cut array
    """

    w = array.size
    ret = None

    if w >= s:
        ret = array[:s]
    else:
        ret = np.pad( array, (0,s-w), constant_values=(0,0) )

    return ret

# %%

# prepare array of pcps

samples2keep = 8000

names = []
categories = []
audios = np.zeros( (len(audio_structs), samples2keep), dtype='float32' )
for i,p in enumerate( audio_structs ):
    print(p)
    names.append( p.name )
    categories.append( p.category )
    if hasattr(p, 'audio'):
        audios[i,:] = pad_end_cut( p.audio, samples2keep )

audio = audios.astype('float32')
# save pickle

with open('data/' + os.sep + 'names.pickle', 'wb') as handle:
    pickle.dump(names, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'categories.pickle', 'wb') as handle:
    pickle.dump(categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

if len(audios) > 0:
    with open('data/' + os.sep + 'audios.pickle', 'wb') as handle:
        pickle.dump(audios, handle, protocol=pickle.HIGHEST_PROTOCOL)