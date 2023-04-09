import librosa
import numpy as np
import matplotlib

def getFile(path):
    sr = librosa.get_samplerate(path)
    audio = librosa.stream(path)