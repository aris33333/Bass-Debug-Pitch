import librosa
import matplotlib.pyplot as plt 
import numpy as np
import pysptk as sp

path = 'sounds/MAIN_OUT.wav'
# Load audio signal
y, sr = librosa.load(path)

f = sp.swipe(data, fs, self.hopsize, min=10, max=600, otype='f0')
