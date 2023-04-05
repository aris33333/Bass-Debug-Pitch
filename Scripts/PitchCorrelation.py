import numpy as np
import librosa
import pysptk as sp
import pandas as pd
import matplotlib.pyplot as plt

class analyzer():       

    def __init__(self):
        self.self
        
    def getData(path, w):

        data, sr = librosa.load(path)
        subsampled = []
        for i in range(0, len(data), w):
            subsampled.append(np.average(data[i:i+w]))
        subsampled = np.array(subsampled)    
        return subsampled, sr

    def getFreq(data, fs, hopsize, threshold):
       
       #Finding RMS energy of each sample
       rms_audio = librosa.feature.rms(y = data, hop_length = hopsize) 
       rms_audio = rms_audio.flatten().tolist()
       rms_audio.pop()
       rms_audio = np.array(rms_audio)

       #Getting frequency values
       f = sp.swipe(data, fs, hopsize, min=10, max=5000, otype='f0')
       #Gating frequency values to remove unecessary frequency data when there is a silence
       f = np.where(rms_audio >= threshold, f, 0)

       #Find total length by muliplying the window width with the size of the array and multiplying it with the sampling period
       total_length = (len(f) * hopsize) * (1 / fs)
       #Creating an array of numberss with fixed windowed sampling intervals
       time = np.arange(0, total_length, hopsize * (1 / fs))

       data = {'f0': f, 
               'Time': time,
               'RMS': rms_audio}
       
       return data, f, time, rms_audio
    
    def subprocessMethod(subprocess_path, w):
        
        data, fs = librosa.load(subprocess_path)
        ctr = 0
        rst = False
        subsampled = []
        
        for i in range(0, len(data), w):
            subsampled = np.average(data[i:i+w])
        subsampled = np.array(subsampled)
        
        for j in range(0, len(subsampled)):
            if rst is False: 
                    if (subsampled[j] * subsampled[j+1] < 0):
                        rst = True
                        ctr = 0
                    else:
                        ctr += ctr
        f = (1/ctr) * (1/fs) 
        return f

    def compareTuners(sub, f):
        pass
        
    def process(clean, dirt, time, mode):

        #Mode if true processes clean as ideal octave values. Mode if false doesn't do anything to the clean signal frequencies
        if mode: 
            clean = clean / 2
        elif not mode:
            clean = clean

        #Calculate pitch deviation from ideal values
        semi = (12 * np.log2(dirt / clean))

        #Remove values that are -inf/inf
        semi = np.where( semi != float('inf'), semi, 0) 
        semi = np.where( semi != - float('inf'), semi, 0)
        cents = semi * 100
        
        setFlag = []
        isOctave = []
        #Setting a flag for unstable values and detecting octave differences
        for i in range(0, len(semi)-1):
            if semi[i] == float('nan'):
                setFlag.append(-1)
            elif semi[i] == 0:
                setFlag.append(0)
            elif semi[i+1] - semi[i] != semi[i]:
                if semi[i] >= semi[i+1] * 0.95 and semi[i] <= semi[i+1] * 1.05:
                    setFlag.append(1) 
                else:
                    setFlag.append(0)
            #Checking for octave differences within bounds and check if there are values over an octave
            if 0.95 * 12 <= semi[i] <= 1.05 * 12 or semi[i] > 12:
                isOctave.append(1)
            else:
                isOctave.append(0)
        setFlag.append(-1) #Ignoring last value
        isOctave.append(0) #Ignoring last value
        
        data = {'Time': time, 
                'Clean': clean, 
                'Dirt': dirt, 
                'Semitones': semi, 
                'Cents': cents,
                'isStable': setFlag,
                'isOctave': isOctave}

        return data, time, clean, dirt, semi, setFlag, isOctave
    
    def plot(time, signal, f=None, sub=None, rms_audio=None, dev=None, flags=None, isOctave=None):
        
        #Scaling data and setting default so data can be plotted if other values are omitted
        signal = signal * 1000
        
        remFlags = False
        remOctave = False
        remRMS = False
        remF = False
        remDev = False
        remSub = False
        
        if flags is None:
            remFlags = True
        else: 
            for i in range(0, len(time)):
                flags[i] = 100 + 10 * flags[i]
                
        if isOctave is None:
            remOctave = True
        else: 
            for i in range(0, len(time)):
                isOctave[i] = 100 + 10 * isOctave[i]
                
        if rms_audio is None:
            remRMS = True
        else: rms_audio = rms_audio * 1000
        
        if sub is None:
            remSub = True
        else: sub
            
        if f is None:
            remF = True
        else: f
        
        if dev is None:
            remDev = True
        else: dev
            
        #Plotting frequency data and the audio clip for visualization
        plt.plot(time, signal)
        legend = ["Signal"]
        
        if not remF: 
            plt.plot(time, f) 
            legend.append("F0")
        else: pass
        
        if not remRMS: 
            plt.plot(time, rms_audio)
            legend.append("RMS")
        else: pass
        
        if not remDev: 
            plt.plot(time, dev)
            legend.append("Deviation in Semitones")
        else: pass
        
        if not remFlags:
            plt.plot(time, flags)
            legend.append("Flags")
        else: pass
        
        if not remOctave:
            plt.plot(time, isOctave)
            legend.append("Octave and Greater Differences")
        else: pass
        
        if not remSub:
            plt.plot(time, sub)
            legend.append("Sub Combined Frequency")
        else: pass
        
        plt.legend(legend, loc = "upper right")
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
sub_file = 'sounds/SUB_COMBINED_peak_cleaner.wav'

test_output = 'freq.csv'
dir_output = 'octave.csv'

octave, sr = analyzer.getData(octaver_file, 16)
clean, sr = analyzer.getData(clean_file, 16)
sub_freq = analyzer.subprocessMethod(sub_file, 16)

#Args for getFreq method: data, sampling frequency, hopsize, and threshold for gating frequency values
#Clean
clean_data, clean_freq, time, rms_clean = analyzer.getFreq(clean, sr, 1,  0.00000005)
#Dirt
octave_data, octave_freq, time_octave, rms_dirt = analyzer.getFreq(octave, sr, 1, 0.00000005)

print(f"Min RMS: {np.min(rms_clean)} Max RMS: {np.max(rms_clean)} Mean RMS of the clip: {np.mean(rms_clean)} in Clean")
print(f"Min RMS: {np.min(rms_dirt)} Max RMS: {np.max(rms_dirt)} Mean RMS of the clip: {np.mean(rms_dirt)} in Dirt")

#True = OCTAVER, False = SYNTH
processor_data, time, clean, dirt, semi, flags, isOctave = analyzer.process(clean_freq, octave_freq, time, True)

#Pitch Correlation Function and Frequency Data
writeData.writeList(test_output, octave_data)
writeData.writeList(dir_output, processor_data)

analyzer.plot(time, octave, octave_freq, sub_freq, None, semi, flags, isOctave)