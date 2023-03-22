import librosa

def find_normalized_silences(audio_file, threshold=0.05, min_duration=0.1):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Compute the short-time energy of the audio signal
    frame_length = int(sr * 0.02) # 20ms
    hop_length = int(sr * 0.01) # 10ms
    energy = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length)

    # Convert the energy to decibels
    energy_db = librosa.amplitude_to_db(energy)

    print('energy_db:', energy_db)

    # Find segments of silence in the audio
    silence_segments = librosa.effects.split(y, top_db=threshold)

    print('silence_segments:', silence_segments)

    # Filter out segments that are too short
    min_duration_samples = int(min_duration * sr)
    silence_segments = [(start, end) for start, end in silence_segments if end - start >= min_duration_samples]

    print('filtered_segments:', silence_segments)

    # Normalize the segments of silence
    normalized_segments = []
    for start, end in silence_segments:
        center = int((start + end) / 2)
        length = end - start
        normalized_start = max(center - int(length / 2), 0)
        normalized_end = min(center + int(length / 2), len(y))
        normalized_segments.append((normalized_start, normalized_end))

    # Return the list of normalized silence segments
    return normalized_segments

silences = find_normalized_silences('sounds/MAIN_OUT_WET.wav', threshold=0.0001, min_duration=0.1)
print(silences)
