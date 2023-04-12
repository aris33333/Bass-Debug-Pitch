import numpy as np
import matplotlib.pyplot as plt
import librosa 
import pysptk as sp

signal, sr = librosa.load('sounds/segments/MAIN_OUT_SEGMENT.wav')

# Define the parameters
window_size = 64          # Window size in samples
hop_size = 16             # Hop size in samples
frequencies = sp.swipe(signal, sr, hop_size, min=10, max=600, otype='f0')

# Compute the STFT
stft = librosa.stft(signal, n_fft=window_size, hop_length=hop_size)

# Find the indices of the frequencies of interest in the frequency axis
f = librosa.fft_frequencies(sr=len(signal), n_fft=window_size)
freq_idxs = [np.argmin(np.abs(f - freq)) for freq in frequencies]

# Extract the magnitude and phase information for the frequencies of interest
magnitudes = np.abs(stft[freq_idxs, :])
phases = np.angle(stft[freq_idxs, :])

# Plot the signal, magnitude, and phase information
fig, axs = plt.subplots(nrows=3, sharex=True)

# Plot the signal
axs[0].plot(np.linspace(0, 1, len(signal)), signal)
axs[0].set_ylabel('Signal')

# Plot the magnitude information
axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[0, :])), sr=len(signal), hop_length=hop_size), magnitudes[0, :], label='{} Hz'.format(frequencies[0]))
axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[1, :])), sr=len(signal), hop_length=hop_size), magnitudes[1, :], label='{} Hz'.format(frequencies[1]))
axs[1].set_ylabel('Magnitude (log scale)')

# Plot the phase information
axs[2].plot(librosa.frames_to_time(np.arange(len(phases[0, :])), sr=len(signal), hop_length=hop_size), np.degrees(phases[0, :]), label='{} Hz'.format(frequencies[0]))
axs[2].plot(librosa.frames_to_time(np.arange(len(phases[1, :])), sr=len(signal), hop_length=hop_size), np.degrees(phases[1, :]), label='{} Hz'.format(frequencies[1]))
axs[2].set_ylim(-180, 180)
axs[2].set_yticks(np.arange(-180, 181, 90))
axs[2].set_ylabel('Phase (degrees)')

axs[-1].set_xlabel('Time (s)')

plt.grid()
plt.show()
