import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

class AudioInfo:
    
    maximum_columns = 0
    
    def __init__(self, filepath, sr=44100, keep_audio=False):
        if filepath.split('.')[-1] in ['wav', 'mp3']:
            self.name = filepath.split('.')[-2].split(os.sep)[-1]
            self.audio, sr = librosa.load( filepath, sr=sr )
            self.sr = sr
            self.extract_power_spectrum()
            if AudioInfo.maximum_columns < self.power_spectrum.shape[1]:
                AudioInfo.maximum_columns = self.power_spectrum.shape[1]
            self.define_category()
            self.make_features()
            self.assign_category()
            if not keep_audio:
                del self.audio
        else:
            print('bad format')
    # end __init__

    def extract_power_spectrum(self):
        p = librosa.stft(self.audio, n_fft=1024, hop_length=512)
        self.power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    # end extract_power_spectrum
    
    def define_category(self):
        self.category = 'undefined'
    # end define_category
    
    def plot_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    # end plot_spectrum
    
    def plot_save_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    # plot_save_spectrum
    
    def make_features(self):
        self.blurred_spectrum = cv2.resize(self.power_spectrum, dsize=(10,50), interpolation=cv2.INTER_CUBIC)
        self.features = np.reshape( self.blurred_spectrum, (500,1) )
    # end make_features
    
    def plot_blurred_spectrum(self):
        plt.imshow(self.blurred_spectrum, cmap='gray_r', origin='lower')
    # end plot_blurred_spectrum
    
    def assign_category(self):
        if 'kick' in self.name.lower() or 'bass' in self.name.lower():
            self.category = 'kick'
        elif 'snare' in self.name.lower():
            self.category = 'snare'
        elif 'tom' in self.name.lower():
            self.category = 'tom'
        elif 'hat' in self.name.lower() or 'hh' in self.name.lower():
            self.category = 'hihat'
        elif 'bell' in self.name.lower():
            self.category = 'bell'
        elif 'conga' in self.name.lower():
            self.category = 'conga'
        elif 'bongo' in self.name.lower():
            self.category = 'bongo'
        elif 'ride' in self.name.lower():
            self.category = 'ride'
        elif 'ride' in self.name.lower():
            self.category = 'ride'
        elif 'crash' in self.name.lower():
            self.category = 'crash'
        elif 'agogo' in self.name.lower():
            self.category = 'agogo'
        elif 'clave' in self.name.lower():
            self.category = 'clave'
        elif 'clap' in self.name.lower():
            self.category = 'clap'
        elif 'rim' in self.name.lower():
            self.category = 'rim'
        elif 'stick' in self.name.lower():
            self.category = 'stick'
        elif 'tamb' in self.name.lower():
            self.category = 'tambourine'
        elif 'shak' in self.name.lower():
            self.category = 'shaker'
        else:
            self.category = 'unknown'
    
# end class AudioInfo
