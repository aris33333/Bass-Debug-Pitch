import librosa
import matplotlib.pyplot as plt 
import numpy as np

path = 'sounds/MAIN_OUT.wav'
# Load audio signal
y, sr = librosa.load(path)

# Compute the non-silent intervals (i.e., the intervals where the signal is above a certain threshold)
threshold = 6  # adjust this as needed
non_silent_intervals = librosa.effects.split(y, top_db=threshold)

# Create a binary mask to nullify the silent parts
mask = np.zeros_like(y, dtype=bool)
for interval in non_silent_intervals:
    start = interval[0]
    end = interval[1]
    mask[start:end] = True

# Apply the mask to the audio signal
y_masked = y * mask

# Plot the original and masked audio signals
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.title('Original audio signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(y_masked)
plt.title('Masked audio signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


"""# Invert the result to obtain the silent intervals
total_duration = librosa.get_duration(y, sr)
silent_intervals = []
if non_silent_intervals[0][0] > 0:
    silent_intervals.append([0, non_silent_intervals[0][0]])
for i in range(len(non_silent_intervals)-1):
    start = non_silent_intervals[i][1]
    end = non_silent_intervals[i+1][0]
    if end - start > 0:
        silent_intervals.append([start, end])
if non_silent_intervals[-1][1] < total_duration:
    silent_intervals.append([non_silent_intervals[-1][1], total_duration])

# Print the silent intervals
for interval in silent_intervals:
    print(f"Silent interval: {interval[0]:.2f}-{interval[1]:.2f} seconds")"""