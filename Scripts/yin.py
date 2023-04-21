from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot

def f(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return np.sin(x * np.pi * 2 * f_0) * envelope(x)

def ACF(f, W, t, lag):
    return np.sum(
        f[t : t + W] * f[lag + t : lag + t + W]
    )

def DF(f, W, t, lag):
    return ACF(f, W, t, 0) + ACF(f, W, t + lag, 0) - (2 * ACF(f, W, t, lag))

def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1
    return DF(f, W, t, lag) / np.sum([DF(f, W, t, j+1) for j in range(lag)]) * lag

def detect_pitch(f, W, t, fs, bounds, thresh = 0.1):
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < thresh: 
            sample = i + bounds[0]
            break
        if sample is None: 
            sample = np.argmin(CMNDF_vals) + bounds[0]
    return fs / sample

W = 32

data = f(10)
bounds = [10, 500]

detect_pitch(data, W)
