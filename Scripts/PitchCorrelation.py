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
       
       rms = (librosa.feature.rms(y=data, hop_length=hopsize)) 
       rms = rms.flatten().tolist().append(0)
       
       f = sp.swipe(data, fs, hopsize, min=10, max=1000, otype='f0')

       "find total length by muliplying the window width with the size of the array and multiplying it with the sampling period"
       total_length = (len(f) * hopsize) * (1 / fs)
       "creating an array of numberss with fixed windowed sampling intervals"
       time = np.arange(0, total_length, hopsize * (1 / fs))

       data = {'F0': f, 'Time': time, 'RMS': rms}
       
       return data, f, time, rms
    
    def process(clean, dirt, time, mode):

        "Mode if true processes clean as ideal octave values. Mode if false doesn't do anything to the clean signal frequencies"
        if mode: 
            clean = clean / 2
        elif not mode:
            clean = clean

        "Calculate pitch deviation from ideal values"
        semi = (12 * np.log2(dirt / clean))

        "Remove values that are inf"
        semi = np.where( semi != float('inf'), semi, 0) 
        semi = np.where( semi != - float('inf'), semi, 0)
        cents = semi * 100

        data = {'time': time, 'clean': clean, 'dirt': dirt, 'deviation in semitones': semi, 'deviation in cents': cents}

        return data, time, clean, dirt, semi, cents, rms
    
class writeData:
    def __init__(self):
        self.self
    
    def writeList(path, data):
        df = pd.DataFrame(data)
        df.to_csv(path)


clean_file = 'sounds/UIMX-855_total_silence.wav'
octaver_file = 'sounds/MAIN_OUT_WET.wav'

test_output = 'freq.csv'
dir_output = 'octave.csv'

octave, sr = analyzer.getData(octaver_file)
clean, sr = analyzer.getData(clean_file)

"Args for getFreq method: data, sampling frequency, hopsize"
clean_data, clean_freq, time, rms = analyzer.getFreq(clean, sr, 512, 0.00001)
octave_data, octave_freq, time_octave, rms2 = analyzer.getFreq(octave, sr, 512, 0.00001)

"True = OCTAVER, False = SYNTH"
processor_data, time, clean, dirt, semi, cents, rms = analyzer.process(clean_freq, octave_freq, time, True)

"Pitch Correlation Function and Frequency Data"
writeData.writeList(dir_output, processor_data)

"Plotting frequency data and the file itself for visualization"
#plt.plot(time, 1000 * octave)
plt.plot(time, octave_freq)
#plt.plot(time, rms2)
plt.show()
plt.close()

