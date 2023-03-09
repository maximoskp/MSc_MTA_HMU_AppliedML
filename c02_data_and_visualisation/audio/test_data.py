# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

from audio_data_processing import AudioInfo

if not os.path.exists('data'):
    os.makedirs('data')

# %%

file_path = '../../data/drumMachines200/AKAI XE-8/MaxV - XE8 Block 1 ext_14.wav'
test_obj = AudioInfo( file_path , keep_audio=False, keep_aux=False )

print( 'size of data: ' + str(len(pickle.dumps(test_obj, -1))) )