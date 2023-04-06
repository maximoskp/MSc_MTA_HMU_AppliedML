# -*- coding: utf-8 -*-

import os
import pandas as pd
import librosa
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

folder = '../data/TinySOL2020/'
samples = {}
if not os.path.exists('data'):
    os.makedirs('data')

df = pd.read_csv( folder + 'TinySOL_metadata.csv' )

categories = []

# get categories and instruments
for d in os.listdir(folder):
    if os.path.isdir( folder + d ):
        print('category: ', d)
        categories.append( d )

tiny_SOL_A234 = {}

# %%

# load data
tiny_SOL_A2 = {}
for c in categories:
    tiny_SOL_A2[c] = {}
    print('extracting: ' + c)
    paths = df[ (df['Family'] == c) & (df['Pitch'] == 'A2') ]['Path']
    tiny_SOL_A2[c] = []
    tiny_SOL_A234[c] = []
    for p in paths:
        print( p )
        s, _ = librosa.load( folder + p, sr=8000 )
        # keep 1 sec
        tiny_SOL_A2[c].append( s[:8000] )
        tiny_SOL_A234[c].append( s[:8000] )

with open('data/' + os.sep + 'tiny_SOL_A2.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_A2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

# load data
tiny_SOL_A3 = {}
for c in categories:
    tiny_SOL_A3[c] = {}
    print('extracting: ' + c)
    paths = df[ (df['Family'] == c) & (df['Pitch'] == 'A3') ]['Path']
    tiny_SOL_A3[c] = []
    for p in paths:
        print( p )
        s, _ = librosa.load( folder + p, sr=8000 )
        # keep 1 sec
        tiny_SOL_A3[c].append( s[:8000] )
        tiny_SOL_A234[c].append( s[:8000] )

with open('data/' + os.sep + 'tiny_SOL_A3.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_A3, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

# load data
tiny_SOL_A4 = {}
for c in categories:
    tiny_SOL_A4[c] = {}
    print('extracting: ' + c)
    paths = df[ (df['Family'] == c) & (df['Pitch'] == 'A4') ]['Path']
    tiny_SOL_A4[c] = []
    for p in paths:
        print( p )
        s, _ = librosa.load( folder + p, sr=8000 )
        # keep 1 sec
        tiny_SOL_A4[c].append( s[:8000] )
        tiny_SOL_A234[c].append( s[:8000] )

with open('data/' + os.sep + 'tiny_SOL_A4.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_A4, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/' + os.sep + 'tiny_SOL_A234.pickle', 'wb') as handle:
    pickle.dump(tiny_SOL_A234, handle, protocol=pickle.HIGHEST_PROTOCOL)