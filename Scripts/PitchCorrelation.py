import numpy as np
import librosa
import pysptk as sp
import pandas as pd
import matplotlib.pyplot as plt

class analyzer():       

    def __init__(self, window=None, threshold=None, hopsize=None, tolerance=None):
        self.window = window
        self.threshold = threshold
        self.hopsize = hopsize
        self.tolerance = tolerance

    def getData(self, path):
        data, sr = librosa.load(path)

        #Audio can be subsampled, by avereraging it with a window width of W
        subsampled = []
        if self.window is None:
            return data, sr
        else:
            for i in range(0, len(data), self.window):
                subsampled.append(np.mean(data[i:i+self.window]))  
            return np.array(subsampled), sr

    def getFreq(self, data, fs):
        
       # Compute the non-silent intervals (i.e., the intervals where the signal is above a certain threshold)
       non_silent_intervals = librosa.effects.split(data, top_db=self.threshold)

       # Create a binary mask to nullify the silent parts
       mask = np.zeros_like(data, dtype=bool)
       for interval in non_silent_intervals:
            start = interval[0]
            end = interval[1]
            mask[start:end] = True

       #Getting frequency values
       f = sp.swipe(data, fs, self.hopsize, min=10, max=600, otype='f0')
       #Applying the mask
       if self.threshold is not None:
             f = f * mask
       else: f

       #Find total length by muliplying the window width with the size of the array and multiplying it with the sampling period
       total_length = (len(f) * self.hopsize) * (1 / fs)

       #Creating an array of numbers with fixed windowed sampling intervals
       time = np.arange(0, total_length, self.hopsize * (1 / fs))

       data = {'f0': f, 
               'Time': time}
       
       return data, f, time, mask
    
    def subProcess(self, subprocess_path, mask): #Clean Signal 
        #Subprocess method is the square waved signal output. Used to detect timbre changes and an alternative reference for pitch detection.
        #Shows up as an abrupt period change, which reflects in the fundamental frequency.  
        data, fs = librosa.load(subprocess_path)
        ctr = 0
        f = []
        store = 0
        
        #When a product of a sample is negative then the signal must have a sign change, which is when the counter (number of samples before sign change) is reset to 0 and a new count starts.
        #Dividing it with the sampling frequency gives the estimated fundamental frequency.
        for j in range(1, len(data)):
            if (data[j] * data[j-1] < 0):
                if ctr > 1:
                    store = (fs/ctr)/2
                else:
                    store = 0
                ctr = 0
            else:
                ctr += 1 
            f.append(store)
        f.append(f[len(f)-1]) #Since the sub file is being read from the 2nd element, it has one less elment. 
        f = mask * f #Applying gate on the sub process

        if self.window is None:
            return np.array(f)
        else:
            averaged = []
            for i in range(0, len(f), self.window):
                averaged.append(np.mean(f[i:i+self.window]))
            return np.array(averaged)
        
    def processDiff(self, clean, dirt, time, mode):
        #Mode if true processes clean as ideal octave values. Mode if false doesn't do anything to the clean signal frequencies
        if mode: 
            clean = clean/2
        elif not mode:
            clean = clean

        #Calculate pitch deviation from ideal values
        semi = (12 * np.log2(dirt/clean))

        #Remove values that are -inf/inf
        semi = np.where( semi != float('inf'), semi, 0) 
        semi = np.where( semi != - float('inf'), semi, 0)
        cents = semi * 100
        
        setFlag = []
        isOctave = []
        tol = self.tolerance / 10
        #Setting a flag for unstable values and detecting octave differences
        for i in range(0, len(semi)-1):
            if semi[i] == float('nan'):
                setFlag.append(-1)
            elif semi[i] == 0:
                setFlag.append(0)
            elif semi[i+1] - semi[i] != semi[i]:
                if semi[i] >= semi[i+1] * (1 - tol) and semi[i] <= semi[i+1] * (1 + tol):
                    setFlag.append(1) 
                else:
                    setFlag.append(0)

            #Checking for octave differences within bounds and check if there are values over an octave
            if (1 - tol) * 12 <= semi[i] <= (1 + tol) * 12 or semi[i] > 12:
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

        return data, semi, setFlag, isOctave
    
    def plot(self, time, octave, clean=None, f=None, sub=None, dev=None, flags=None, isOctave=None):
        
        #Scaling data and setting defaults so data can be plotted if other values are omitted   
        remFlags = False
        remOctave = False
        remClean = False
        remF = False
        remDev = False
        remSub = False
        
        if flags is None:
            remFlags = True
        else: 
            for i in range(0, len(time)):
                flags[i] = 10 + 10 * flags[i]
                
        if isOctave is None:
            remOctave = True
        else: 
            for i in range(0, len(time)):
                isOctave[i] = 10 + 10 * isOctave[i]
                
        if clean is None:
            remClean = True
        else: clean
        
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
        fig, ax = plt.subplots(3, sharex=True)
        fig.suptitle("Data Correlation")
        ax[0].plot(time, octave)
        ax[0].set_title("Processed Signal")
        ax[0].legend(["Octave Signal"], loc = "upper right")
        ax[0].grid()
       
        legend = []
        if not remF: 
            ax[1].plot(time, f) 
            legend.append("F0")
        else: pass
    
        if not remDev: 
            ax[1].plot(time, dev)
            legend.append("Deviation in Semitones")
        else: pass
        
        if not remFlags:
            ax[1].plot(time, flags)
            legend.append("Flags")
        else: pass
        
        if not remOctave:
            ax[1].plot(time, isOctave)
            legend.append("Octave and Greater Differences")
        else: pass
        
        if not remSub:
            ax[1].plot(time, sub)
            legend.append("Sub Combined Frequency")
        else: pass
        ax[1].legend(legend, loc = "upper right")
        ax[1].set_title("Processed Correlation")
        ax[1].grid()
        
        if not remClean: 
            ax[2].plot(time, clean)
            ax[2].legend(["Clean Signal"], loc="upper right")
            ax[2].set_title("Clean Audio")
            ax[2].grid()
        else: ax[2].set_visible(False)

        plt.tight_layout()
        plt.show()
        
    def spectrum(self, signal, sr, window_size, hop_size, args=None):

        #Estimate Fundamental Frequencies
        frequencies = sp.swipe(signal, sr, hop_size, min=10, max=600, otype='f0')

        #Args for looking at other harmonics (Multiplies to F0)
        if args is None: frequencies
        else: frequencies *= args

        #Compute the STFT
        stft = librosa.stft(signal, n_fft=window_size, hop_length=hop_size)

        #Find the indices of the frequencies of interest in the frequency axis
        f = librosa.fft_frequencies(sr=len(signal), n_fft=window_size)
        freq_idxs = [np.argmin(np.abs(f - freq)) for freq in frequencies]

        #Extract the magnitude and phase information for the frequencies of interest
        magnitudes = np.abs(stft[freq_idxs, :])
        phases = np.angle(stft[freq_idxs, :])

        #Plot the signal, magnitude, and phase information
        fig, axs = plt.subplots(nrows=3, sharex=True)

        #Plot the signal
        axs[0].plot(np.linspace(0, 1, len(signal)), signal)
        axs[0].set_ylabel('Signal')
        axs[0].grid()

        #Plot the magnitude information
        axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[0, :])), sr=len(signal), hop_length=hop_size), magnitudes[0, :], label='{} Hz'.format(frequencies[0]))
        axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[1, :])), sr=len(signal), hop_length=hop_size), magnitudes[1, :], label='{} Hz'.format(frequencies[1]))
        axs[1].set_ylabel('Magnitude (log scale)')
        axs[1].grid()

        #Plot the phase information
        axs[2].plot(librosa.frames_to_time(np.arange(len(phases[0, :])), sr=len(signal), hop_length=hop_size), np.degrees(phases[0, :]), label='{} Hz'.format(frequencies[0]))
        axs[2].plot(librosa.frames_to_time(np.arange(len(phases[1, :])), sr=len(signal), hop_length=hop_size), np.degrees(phases[1, :]), label='{} Hz'.format(frequencies[1]))
        axs[2].set_ylim(-180, 180)
        axs[2].set_yticks(np.arange(-180, 181, 90))
        axs[2].set_ylabel('Phase (degrees)')
        axs[2].grid()

        axs[-1].set_xlabel('Time (s)')

        plt.show()

class writeData:
    def __init__(self):
        pass
    
    def writeList(path, data):
        df = pd.DataFrame(data)
        df.to_csv(path)
        
#Paths
#Audio
clean_file = 'sounds/UIMX-855_total_silence.wav'
octaver_file = 'sounds/MAIN_OUT_WET.wav'
sub_file = 'sounds/SUB_COMBINED_total_silence.wav'

#CSV
test_output = 'freq.csv'
dir_output = 'octave.csv'

#Args: Averaging Window Wdith, Threshold for Gating, Hopsize, Tolerance. If None: Averaging and Gating can be skipped. 
#Init Object
analyzer = analyzer(None, 20, 1, 5)
#Args: File path
octave, sr = analyzer.getData(octaver_file)
clean, sr = analyzer.getData(clean_file)

#Args: Data, Fs.
#Clean
clean_data, clean_freq, time, clean_mask = analyzer.getFreq(clean, sr)
#Dirt
octave_data, octave_freq, time_octave, octave_mask = analyzer.getFreq(octave, sr)

#Sub Combined File Processing
#Args: File Path, Binary Mask
sub_freq = analyzer.subProcess(sub_file, octave_mask)

#Args: Clean Freq, Dirt Freq, Time
#True = OCTAVER, False = SYNTH
processor_data, dev, Flags, isOctave = analyzer.processDiff(clean_freq, octave_freq, time, True)

#Args: Time, Processed Signal, Clean, Processed Frequency, Sub Process Freq, Deviation, Flags, Octave Errors. 
#Use None for omitting data (cannot omit Processed Audio and Time).
analyzer.plot(time, octave, clean, octave_freq, sub_freq, None, None, None)

#Args: Signal Data, Sample Rate, WINDOW LENGTH, HOP SIZE
analyzer.spectrum(octave, sr, 1024, 512, None)