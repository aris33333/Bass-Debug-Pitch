import numpy as np
import librosa
import matplotlib.pyplot as plt

audio_file, sr = librosa.load('sounds/segments/SUB_COMBINED.wav', sr=None)

fft = np.fft.fft(audio_file)
freqs = np.fft.fftfreq(len(audio_file), 1/sr)

#Find the fundamental frequency, assuming peak in the fft is f0
fundamental_idx = np.argmax(np.abs(fft))
fundamental_freq = freqs[fundamental_idx]

#Harmonics are multiples of i
harmonic_freqs = [fundamental_freq * i for i in range(1, 4)]

#Find decay of each harmonic
decay_rates = []
for freq in harmonic_freqs:
    harmonic_idx = np.where(np.abs(freqs - freq) < 10)[0][0]
    start_amp = np.abs(fft[harmonic_idx])
    end_amp = np.abs(fft[harmonic_idx + sr//2])  # measure decay after 0.5 seconds
    decay_rates.append(end_amp / start_amp)

#Plotting stuff
fig, ax = plt.subplots()
ax.bar(np.arange(len(harmonic_freqs)), decay_rates)
ax.set_xticks(np.arange(len(harmonic_freqs)))
ax.set_xticklabels([f'{freq:.1f} Hz' for freq in harmonic_freqs])
ax.set_xlabel('Harmonic frequency')
ax.set_ylabel('Decay rate')
ax.set_title('Decay of first three harmonics for audio signal')
plt.show()
