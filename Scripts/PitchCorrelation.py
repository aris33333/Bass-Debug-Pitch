import numpy as np
import librosa
import pysptk as sp
import pandas as pd
import matplotlib.pyplot as plt

class analyzer():       

    def __init__(self):
        self.self
        
    def getData(path):

        data, sr = librosa.load(path)
        return data, sr

    def getFreq(data, fs, hopsize, threshold):
       
       "Finding RMS energy of each sample"
       rms_audio = librosa.feature.rms(y = data, hop_length = hopsize) 
       rms_audio = rms_audio.flatten().tolist()
       rms_audio.pop()
       rms_audio = np.array(rms_audio)

       "Getting frequency values"
       f = sp.swipe(data, fs, hopsize, min=10, max=1000, otype='f0')
       "Gating frequency values to remove unecessary frequency data when there is a silence"
       f = np.where(rms_audio >= threshold, f, 0)

       "find total length by muliplying the window width with the size of the array and multiplying it with the sampling period"
       total_length = (len(f) * hopsize) * (1 / fs)
       "creating an array of numberss with fixed windowed sampling intervals"
       time = np.arange(0, total_length, hopsize * (1 / fs))

       data = {'f0': f, 
               'Time': time,
               'RMS': rms_audio}
       
       return data, f, time, rms_audio
    
    def subprocessMethod(subprocess_path):
        
        sqr, fs = librosa.load(subprocess_path)
        max = np.max(sqr)
        min = np.min(sqr)
        count = 0 
    
    def compareTuners(subprocess, f):
        pass
        
        
    def process(clean, dirt, time, mode):

        "Mode if true processes clean as ideal octave values. Mode if false doesn't do anything to the clean signal frequencies"
        if mode: 
            clean = clean / 2
        elif not mode:
            clean = clean

        "Calculate pitch deviation from ideal values"
        semi = (12 * np.log2(dirt / clean))

        "Remove values that are -inf/inf"
        semi = np.where( semi != float('inf'), semi, 0) 
        semi = np.where( semi != - float('inf'), semi, 0)
        cents = semi * 100
        
        setFlag = []
        
        "Setting a flag for unstable values"
        for currrentVal in semi[0:len(semi)]:
            for previousVal in semi[-1:len(semi)]:
                if currrentVal == float('nan'):
                    setFlag = 'Ignore'
                elif currrentVal == 0:
                    setFlag = 'True'
                elif currrentVal > 0:
                    if currrentVal > previousVal * 0.95 and currrentVal < previousVal * 1.05:
                        setFlag = 'True'
                    else:
                        setFlag = 'False'

        data = {'Time': time, 
                'Clean': clean, 
                'Dirt': dirt, 
                'Deviation in semitones': semi, 
                'Deviation in cents': cents,
                'Flags for stability': setFlag}

        return data, time, clean, dirt, semi
    
    def plot(time, signal, f=None, rms_audio=None, dev=None):
        
        "Scaling data and setting default so data can be plotted if other values are omitted"
        signal = signal * 1000
        
        if rms_audio is None:
            rms_audio = np.zeros(len(time))
        else: rms_audio = rms_audio * 1000
            
        if f is None:
            f = np.zeros(len(time))
        else: f
        
        if dev is None:
            dev = np.zeros(len(time))
        else: dev
            
        "Plotting frequency data and the audio clip for visualization"
        plt.plot(time, signal)
        plt.plot(time, f)
        plt.plot(time, rms_audio)
        plt.plot(time, dev)
        plt.legend(["Signal", "f0", "RMS", "Deviation in Semitones"], loc = "upper right")
        plt.show()
        plt.close()

class writeData:
    def __init__(self):
        self.self
    
    def writeList(path, data):
        df = pd.DataFrame(data)
        df.to_csv(path)

clean_file = 'sounds/UIMX-862_peak_cleaner_dirty_transients.wav'
octaver_file = 'sounds/MAIN_OUT.wav'

test_output = 'freq.csv'
dir_output = 'octave.csv'

octave, sr = analyzer.getData(octaver_file)
clean, sr = analyzer.getData(clean_file)

"Args for getFreq method: data, sampling frequency, hopsize, and threshold for gating frequency values"
clean_data, clean_freq, time, rms_clean = analyzer.getFreq(clean, sr, 1, 0.0085)
octave_data, octave_freq, time_octave, rms_dirt = analyzer.getFreq(octave, sr, 1, 0.0085)

"True = OCTAVER, False = SYNTH"
processor_data, time, clean, dirt, semi = analyzer.process(clean_freq, octave_freq, time, True)

"Pitch Correlation Function and Frequency Data"
writeData.writeList(test_output, octave_data)
writeData.writeList(dir_output, processor_data)

analyzer.plot(time, octave, octave_freq, None, semi)