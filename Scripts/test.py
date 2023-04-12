import numpy as np
import matplotlib.pyplot as plt
import librosa 
import pysptk as sp

y, sr = librosa.load('sounds/segments/SUB_COMBINED.wav')

WINDOW = 1024
HOPSIZE = 512

magnitude = []
phase = []

f = sp.swipe(y, sr, 512, min=10, max=600, otype='f0')
freq = librosa.fft_frequencies(sr=sr, n_fft = WINDOW)
yf = librosa.stft(y, hop_length=HOPSIZE, win_length=WINDOW)
    
for i in range(0, len(f)):
 
    index, = np.where(np.isclose(freq, f[i], atol=1/(1/sr * len(y))))
    # Get magnitude and phases
    magnitude.append(np.abs(yf[index]))
    phase.append(np.angle(yf[index]))

total_length = (len(f) * HOPSIZE) * (1 / sr)
time = np.arange(0, total_length, HOPSIZE * (1 / sr))
print(len(time), np.shape(yf))

# Plot a spectrum 
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(time, magnitude) # in a conventional form
ax[0].set_title("Magnitude")
ax[1].plot(time, phase)
ax[1].set_title("Phase")
plt.grid()
plt.show()