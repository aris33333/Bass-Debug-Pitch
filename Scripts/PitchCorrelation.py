import numpy as np
import librosa
import pysptk as sp
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt

class analyzer:       

    def __init__(self, window=None, threshold=None, hopsize=None, tolerance=None):
        self.window = window
        self.threshold = threshold
        self.hopsize = hopsize
        self.tolerance = tolerance
        self.mask = 0

    def getData(self, path, args):
        data, sr = librosa.load(path, sr = 48000)

        #Audio can be subsampled, by avereraging it with a window width of W
        subsampled = []
        if args is False:
            return data, sr
        else:
            for i in range(0, len(data), self.window):
                subsampled.append(np.mean(data[i:i+self.window]))  
            return np.array(subsampled), sr

    def getFreq(self, data, fs):
       #Getting frequency values
       f = sp.swipe(data, fs, self.hopsize, min=10, max=600, otype='f0')

       if self.threshold is None: 
            pass
       else: 
            #Compute the non-silent intervals (i.e., the intervals where the signal is above a certain threshold)
            non_silent_intervals = librosa.effects.split(data, top_db=self.threshold)

            #Create a binary mask to nullify the silent parts
            self.mask = np.zeros_like(data, dtype=bool)
            for interval in non_silent_intervals:
                        start = interval[0]
                        end = interval[1]
                        self.mask[start:end] = True
                
            #Applying the mask        
            f = f * self.mask
      
       #Find total length by muliplying the window width with the size of the array and multiplying it with the sampling period
       total_length = (len(f) * self.hopsize) * (1 / fs)

       #Creating an array of numbers with fixed windowed sampling intervals
       time = np.arange(0, total_length, self.hopsize * (1 / fs))

       data = {'f0': f, 
               'Time': time}
       
       return data, f, time
    
    def subProcess(self, subprocess_path): 
        #Clean Signal 
        #Subprocess method is the square waved signal output. Used to detect timbre changes and an alternative reference for pitch detection.
        #Shows up as an abrupt period change, which reflects in the fundamental frequency.  
        data, fs = librosa.load(subprocess_path)
        ctr = 0
        f = []
        store = 0
        
        #When a product of a sample is negative then the signal must have a sign change,
        #which is when the counter (number of samples before sign change) is reset to 0 and a new count starts.
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
        
        if self.threshold is None: 
            pass
        else: 
            f = self.mask * f #Applying gate on the sub process

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
                if semi[i] >= semi[i+1] * (1 - tol) or semi[i] <= semi[i+1] * (1 + tol):
                    setFlag.append(1) 
                else:
                    setFlag.append(0)
            #Checking for octave differences within bounds and check if there are values over an octave
            if (1 - tol) * 12 >= semi[i] or (1 + tol) * 12 <= semi[i] or semi[i] > 12:
                isOctave.append(0)
            else:
                isOctave.append(1)
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
            
        #Plotting frequency data and the audio clip for visualization
        fig, ax = plt.subplots(3, sharex=True)
        fig.suptitle("Data Correlation")
        ax[0].plot(time, octave)
        ax[0].set_title("Processed Signal")
        ax[0].grid()
       
        legend = []
        if f is not None: 
            ax[1].plot(time, f) 
            legend.append("Processed F0")
        else: pass
    
        if dev is not None: 
            ax[1].plot(time, dev)
            legend.append("Deviation in Semitones")
        else: pass
        
        if flags is not None:
            for i in range(0, len(time)):
                flags[i] = 10 + 10 * flags[i]
            ax[1].plot(time, flags)
            legend.append("Flags")
        else: pass
        
        if isOctave is not None:
            for i in range(0, len(time)):
                isOctave[i] = 10 + 10 * isOctave[i]
            ax[1].plot(time, isOctave)
            legend.append("Octave and Greater Differences")
        else: pass
        
        if sub is not None:
            ax[1].plot(time, sub)
            legend.append("Clean F0/Sub Combined Frequency")
        else: pass
        ax[1].legend(legend, loc = "upper right")
        ax[1].set_title("Processed Correlation")
        ax[1].grid()
        
        if clean is not None: 
            ax[2].plot(time, clean)
            ax[2].set_title("Clean Signal")
            ax[2].grid()
        else: ax[2].set_visible(False)

        plt.tight_layout()
        plt.show()

    def runOctaver(self, exe_path, folder, filename, args, dryrun):
        #Runs the batch processor executable with which runs the audio
        #processing and creates audio files of all debug streams.

        if not args:
            args = ['1']
        if '.wav' in filename:
            raise Exception('Filenames should be given without file extensions')
        # find executable based on our system
        if not os.path.isfile(exe_path):
            raise Exception(f'Executable not found: "{exe_path}"')
        args = [exe_path, f'sounds/clean/{folder}/', filename] + args 
        print(' '.join(args))
        if dryrun:
            return
        output = subprocess.run(args, capture_output=True)
        if output.returncode != 0:
            print(' '.join(output.args))
            raise Exception(output.stderr)
        
    def spectrum(self, signal, sr, window_size, hop_size, args=None):

        #Estimated Fundamental Frequencies
        frequencies = sp.swipe(signal, sr, hop_size, min=10, max=600, otype='f0')

        #Args for looking at other harmonics (Multiplies to F0)
        if args is None: frequencies
        else: frequencies *= args

        stft = librosa.stft(signal, n_fft=window_size, hop_length=hop_size, center=False, window='hann')

        #Find the indices of the frequencies of interest in the frequency axis
        f = librosa.fft_frequencies(sr=len(signal), n_fft=window_size)
        freq_idxs = [np.argmin(np.abs(f - freq)) for freq in frequencies]

        #Extract the magnitude and phase information for the frequencies of interest
        magnitudes = np.abs(stft[freq_idxs, :])
        phases = np.angle(stft[freq_idxs, :])

        fig, axs = plt.subplots(nrows=3, sharex=True)
        axs[0].plot(np.linspace(0, 1, len(signal)), signal)
        axs[0].set_ylabel('Signal')
        axs[0].grid()

        axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[0, :])), sr=len(signal), hop_length=hop_size), magnitudes[0, :], label='{} Hz'.format(frequencies[0]))
        axs[1].semilogy(librosa.frames_to_time(np.arange(len(magnitudes[1, :])), sr=len(signal), hop_length=hop_size), magnitudes[1, :], label='{} Hz'.format(frequencies[1]))
        axs[1].set_ylabel('Magnitude (log scale)')  
        axs[1].grid()

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

############################ INIT ###############################
file = 'PIEZO'
mode = 'SYNTH'
folder = 'open_strings'

#Args: Averaging Window Width, Threshold for Gating, Hopsize, Tolerance. If None: Averaging and Gating can be skipped. 
#Init Object
analyzer = analyzer(None, 20, 1, 5)

#Run Octaver exe and generate data
#exe_path = f'exe/hybrid_octaver_batch_processor_{mode}.exe'
#analyzer.runOctaver(exe_path, folder, file, False, False)

#Audio
clean_file = f'sounds/clean/{folder}/{file}.wav'
if mode == 'SYNTH':
    processed_file = f'sounds/clean/{folder}/{file}_{mode}.wav'
else:
    processed_file = f'sounds/clean/{folder}/processed_{mode}/{file}/MAIN_OUT.wav'
#sub_file = f'sounds/clean/{folder}/processed_{mode}/{file}/SUB_COMBINED.wav'

##################### MAIN PROCESSING ###########################

#Args: File path
prc, sr = analyzer.getData(processed_file, False)
clean, sr = analyzer.getData(clean_file, False)

#Args: Data, Fs.
#Clean
clean_data, clean_freq, time = analyzer.getFreq(clean, sr)
#Dirt
prc_data, prc_freq, time_octave = analyzer.getFreq(prc, sr)

#Sub Combined File Processing
#Args: File Path
#sub, sr = analyzer.getData(sub_file)
#sub_freq = analyzer.subProcess(sub_file)

#Args: Clean Freq, Dirt Freq, Time
#True = OCTAVER, False = SYNTH
processor_data, dev, flags, isOctave = analyzer.processDiff(clean_freq, prc_freq, time, False)

#Args: Time, Processed Signal, Clean, F0-1, F0-2, Deviations, Flags, Octave Errors. 
#Use None for omitting data (cannot omit Processed Audio and Time).
analyzer.plot(time, prc, clean, prc_freq, clean_freq, dev, flags, isOctave)
