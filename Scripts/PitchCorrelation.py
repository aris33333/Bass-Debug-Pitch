import numpy as np
import librosa
import pysptk as sp
import pandas as pd
import matplotlib.pyplot as plt

class analyzer():       

    def __init__(self):
        self.data = data
        
    def getData(path):
        data, sr = librosa.load(path)
        return data, sr

    def getFreq(data, fs, hopsize, threshold):
       
       rms = (librosa.feature.rms(y=data, hop_length=hopsize))  
       rms = np.array([rms])

       f0 = sp.swipe(data, fs, hopsize, min=10, max=1000, otype='f0')
       f0 = np.array([f0])

       "find total length by muliplying the window width with the size of the array and multiplying it with the sampling period"
       total_length = (len(f0) * hopsize) * (1 / fs)
       "creating an array of numberss with fixed windowed sampling intervals"
       time = np.arange(0, total_length, hopsize * (1 / fs))

       """if (rms > threshold).any(): 
           f0 = np.zeros_like(f0)
       else: 
           f0 = f0"""

       return f0, time, rms
    
    def process(clean, dirt, time, mode):

        "Mode if true processes clean as ideal octave values. Mode if false doesn't do anything to the clean signal frequencies"
        if mode: 
            clean = clean / 2
        elif not mode:
            clean = clean

        "Makes data as numpy arrays because sometimes it acts weird"
        clean = np.array(clean)
        dirt = np.array(dirt)

        "Calculate pitch deviation from ideal values"
        dev = (12 * np.log2(dirt / clean))

        "Remove values that are inf"
        dev = np.where( dev != float('inf'), dev, 0) 
        dev = np.where( dev != - float('inf'), dev, 0)
        cents = dev * 100

        data = {'time': time, 'clean': clean, 'dirt': dirt, 'deviation in semitones': dev, 'deviation in cents': cents}

        return data, time, clean, dirt, dev, cents
    
class writeData:
    def __init__(self, path, data1, data2):
        self.path = path
        self.data1 = data1
        self.data2 = data2

    def writeData(path, data1, data2):
        data = {'data1': data2, 'data2': data1}
        df = pd.DataFrame(data)
        df.to_csv(path)
    
    def writeList(path, data1):
        df = pd.DataFrame(data1)
        df.to_csv(path)


clean_file = 'Scripts/sounds/UIMX-855_total_silence.wav'
output = 'Scripts/freq.csv'

octaver_file = 'Scripts/sounds/MAIN_OUT_WET.wav'
output2 = 'Scripts/octave.csv'

octave, sr = analyzer.getData(octaver_file)
clean, sr = analyzer.getData(clean_file)

t_path = 'Scripts/test.csv'

"Args for getFreq method: data, sampling frequency, hopsize"
clean_freq, time, rms = analyzer.getFreq(clean, sr, 1, 0.00001)
octave_freq, time2, rms2 = analyzer.getFreq(octave, sr, 1, 0.00001)

writeData.writeData(t_path, time2, rms2)

"True = OCTAVER, False = SYNTH"
data, time3, clean, dirt, dev, cents = analyzer.process(clean_freq, octave_freq, time, True)

"Raw frequency data"
writeData.writeData(output, clean_freq, octave_freq)

"Pitch Correlation Function Data"
writeData.writeList(output2, data)

"Plotting frequency data and the file itself for visualization"
plt.plot(time, 1000 * octave)
plt.plot(time, octave_freq)
plt.show()
plt.close()

