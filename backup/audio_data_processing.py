import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

class AudioInfo:
    def __init__(self, filepath, sr=44100, n_fft=1024, hop_length=512, keep_audio=False):
        if filepath.split('.')[-1] in ['wav', 'mp3']:
            self.name = filepath.split('.')[-2].split(os.sep)[-1]
            self.audio, sr = librosa.load( filepath, sr=sr )
            self.sr = sr
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.extract_power_spectrum()
            self.secs2keep = 0.5
            self.freqsUpperLimit = 5000
            self.compute_rows_columns()
            self.define_category()
            self.make_features()
            self.assign_category()
            if not keep_audio:
                del self.audio
        else:
            print('bad format')
    # end __init__

    def extract_power_spectrum(self):
        p = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        self.power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    # end extract_power_spectrum
    
    def compute_rows_columns(self):
        self.rows2keep = int( np.ceil( (self.freqsUpperLimit/self.sr)*self.n_fft ) )
        self.columns2keep = int( np.ceil( self.secs2keep/(self.n_fft/self.sr) ) )
    # end compute_rows_columns
    
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
        # centroid
        c = librosa.feature.spectral_centroid(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.centroid = c[0]
        self.mean_centroid = np.mean( self.centroid )
        self.std_centroid = np.std( self.centroid )
        # bandwidth
        b = librosa.feature.spectral_bandwidth(self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.bandwidth = b[0]
        self.mean_bandwidth = np.mean( self.bandwidth )
        self.std_bandwidth = np.std( self.bandwidth )
        self.features = np.reshape( [self.mean_centroid, self.std_centroid, self.mean_bandwidth, self.std_bandwidth], (4,1) )
    # end make_features
    
    def make_features_spectrum(self):
        self.spectrum_part = np.zeros( (self.rows2keep, self.columns2keep) )
        r = min( self.rows2keep, self.power_spectrum.shape[0] )
        c = min( self.columns2keep, self.power_spectrum.shape[1] )
        self.spectrum_part[:r, :c] = self.power_spectrum[:r, :c]
        # print(self.spectrum_part.shape)
        # self.blurred_spectrum = cv2.resize(self.spectrum_part, dsize=(c//3,r//3), interpolation=cv2.INTER_CUBIC)
        # self.features = np.reshape( self.blurred_spectrum, (self.blurred_spectrum.size,1) )
        self.features = np.reshape( self.spectrum_part, (self.spectrum_part.size,1) )
    # end make_features
    
    def plot_spectrum_part(self):
        plt.imshow(self.spectrum_part, cmap='gray_r', origin='lower')
    # end plot_spectrum_part
    
    # def plot_blurred_spectrum(self):
    #     plt.imshow(self.blurred_spectrum, cmap='gray_r', origin='lower')
    # # end plot_blurred_spectrum
    
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
