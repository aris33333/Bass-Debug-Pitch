import soundfile as sf
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

#
# Hardware emulation
#

class Envelope_detector:
    # emulates an op-amp based envelope detector

    def __init__(self, samplerate):
        self.resistance = 220e3
        self.capacitance = 100e-9
        self.period = 1/samplerate
        self.decay_rate = np.exp(-self.period/(self.resistance*self.capacitance))
        self.envelope_signal = 0
        
    def process(self, input_signal):
        if input_signal >= self.envelope_signal:
            self.envelope_signal = input_signal
        else:
            self.envelope_signal = self.decay_rate * self.envelope_signal
            
        return self.envelope_signal
    
    
class Slope_polarity_detector:
    # emulates a comparator-based slope polarity detector 
    
    def __init__(self):
        self.previous_sample = 0
        self.slope_polarity = False
    
    def process(self, input_signal):
        # returns True if the input signal slope is downwards
        if input_signal > self.previous_sample:
            self.slope_polarity = False
        else:
            self.slope_polarity = True
        self.previous_sample = input_signal
        
        return self.slope_polarity
    
    
class Rising_edge_interrupt:
    # emulates a microcontroller GPIO interrupt logic
    
    def __init__(self):
        self.input_previous_state = False
        
    def process(self, input_signal):
        if input_signal == True and self.input_previous_state == False:
            self.input_previous_state = input_signal
            return True
        else:
            self.input_previous_state = input_signal
            return False

        
class Period_doubler():
    # emulates a circuit of two flip flops, connected so that the output
    # period is twice of the input (set/reset cycle) period
    
    def __init__(self):
        self.first_ff_state = False
        
        # second flipflop
        self.clock = False
        self.data = False
        self.q = False
        self.antiq = True
        
    def set(self):
        self.first_ff_state = True
        
    def reset(self):
        self.first_ff_state = False
        
    def process(self):
        if self.clock == False and self.first_ff_state == True:
            self.clock = self.first_ff_state
            self.data = self.antiq

            if self.clock:
                if self.data:
                    self.q = True
                    self.antiq = False
                else:
                    self.q = False
                    self.antiq = True
        else:
            self.clock = self.first_ff_state
            self.data = self.antiq

        return self.q

    
def bandpass_filter(data, order, lowcut, highcut, samplerate):
    coeffs = signal.butter(order, [lowcut, highcut], btype="band", analog=False,
                           output="sos", fs=samplerate)
    return signal.sosfiltfilt(coeffs, data)
    
#
# Actual software classes start here
#
        
class Peak:
    # for storing peak values
    def __init__(self, height, gap):
        self.height = height
        self.gap = gap
        
        
class Peak_tracker:
    def __init__(self):
        # The class keeps track of:
        #   - the height of the previous peak
        #   - the distance to the previous peak
        #   - the distance between two previous peaks
        
        #    | peak gap |
        #
        #    _          _         <-
        #   / \        / \           peak height
        # --   \   ----   \   /-  <- 
        #       \ /        \ /
        #        ¨          ¨
        # 


        self.peak = Peak(0, 0)         

        # how many % of the peak gap has to pass before a new peak is accepted
        self.minimum_wait = 0.5

        self.wait_counter = 0
        self.time_since_previous_peak = 0
        
    def process(self, sample, i, interrupt):
        if interrupt:
            # a signal peak has triggered an interrupt
            peak = Peak(sample, self.time_since_previous_peak)
            
            if peak.height > 2 * self.previous_height():
                # The peak was so high that a string was probably plucked,
                # let's forget the previous gap since the pitch might have
                # changed
                
                # print(i, "height")
                peak = Peak(sample, 1)
                self.add_peak(peak)

            elif self.wait_counter == 0:
                # Enought time has passed since last peak

                # print(i, "wait")
                self.add_peak(peak)

            elif peak.height >= self.previous_height():
                # The new peak is higher than the previous one, so the previous
                # peak was only a local maximum point.

                # print(i, "replace")
                self.replace_peak(peak)

            else:
                # print(i, "miss")
                pass
            
        else:
            # no interrupt, but increment/decrement the counters
            if self.wait_counter > 1:
                self.wait_counter -= 1
            else:
                self.wait_counter = 0
            
            self.time_since_previous_peak += 1
        
        # if time_since_previous_peak = 0, a true peak was found
        return self.time_since_previous_peak, self.wait_counter
    
    def add_peak(self, peak):
        self.peak = peak
    
        self.wait_counter = peak.gap * self.minimum_wait
        self.time_since_previous_peak = 0
        
    def replace_peak(self, peak):
        peak.gap = self.gap_to_previous()
        self.add_peak(peak)
        
    def gap_to_previous(self):
        return self.peak.gap
        
    def previous_height(self):
        return self.peak.height    


class Data_collection:
    # for logging and plotting
    
    def __init__(self, clean_audio):
        self.clean_audio = clean_audio
        self.audio_length = self.clean_audio.size
        
        # create a bunch of numpy arrays for holding the signals
        self.top_signals = {"envelope_signal": np.empty(self.audio_length),
                            "slope_polarity": np.empty(self.audio_length),
                            "interrupt": np.empty(self.audio_length),
                            "tracker": np.empty(self.audio_length),}
    
        self.bottom_signals = {"envelope_signal": np.empty(self.audio_length),
                               "slope_polarity": np.empty(self.audio_length),
                               "interrupt": np.empty(self.audio_length),
                               "tracker": np.empty(self.audio_length),}
        
        self.both_sides = {"top": self.top_signals,
                           "bottom": self.bottom_signals}
        
        self.square_wave = np.empty(self.audio_length)
    
    def log_data(self, polarity, data_type, index, data):
        self.both_sides[polarity][data_type][index] = data
        
    def log_square_wave(self, index, state):
        self.square_wave[index] = state
        
    def get_square_wave(self):
        return self.square_wave
        
    def plot(self):
        
        plt.rcParams['figure.figsize'] = [15, 7]
        plt.tight_layout()
        
        plt.plot(self.clean_audio)

        # top side plots
        plt.plot(self.both_sides["top"]["envelope_signal"])
        top_peak_tracker_scaled = self.both_sides["top"]["tracker"]/np.max(self.both_sides["top"]["tracker"])*np.max(np.abs(self.clean_audio))
        plt.plot(top_peak_tracker_scaled, color="purple", alpha=0.7)
        plt.fill_between(range(self.audio_length), 0, 1*np.max(np.abs(self.clean_audio)), where=(self.both_sides["top"]["interrupt"] > 0), color="green", alpha=0.5)

        # bottom side plots, notice the negative signs
        plt.plot(-self.both_sides["bottom"]["envelope_signal"])
        bottom_peak_tracker_scaled = self.both_sides["bottom"]["tracker"]/np.max(self.both_sides["bottom"]["tracker"])*np.max(np.abs(self.clean_audio))
        plt.plot(-bottom_peak_tracker_scaled, color="purple", alpha=0.7)        
        plt.fill_between(range(self.audio_length), 0, -1*np.max(np.abs(self.clean_audio)), where=(self.both_sides["bottom"]["interrupt"] > 0), color="red", alpha=0.5)
        
        # square wave
        plt.fill_between(range(self.audio_length), 0, 1*np.max(np.abs(self.clean_audio)), where=(self.square_wave == True), color="black", alpha=0.1)
        plt.fill_between(range(self.audio_length), 0, -1*np.max(np.abs(self.clean_audio)), where=(self.square_wave == False), color="black", alpha=0.1)
        
        plt.show()

        



input_file = "sounds/Loop.wav"
output_file = "sounds/outputs/Loop.wav"

clean_audio, samplerate = sf.read(input_file)

filtered_audio = bandpass_filter(clean_audio, 2, 10, 1000, samplerate)

# hardware emulators
top_envelope_detector = Envelope_detector(samplerate)
bottom_envelope_detector = Envelope_detector(samplerate)

top_slope_detector = Slope_polarity_detector()
bottom_slope_detector = Slope_polarity_detector()

top_interrupt = Rising_edge_interrupt()
bottom_interrupt = Rising_edge_interrupt()

period_doubler = Period_doubler()
# hardware emulators end

top_peak_tracker = Peak_tracker()
bottom_peak_tracker = Peak_tracker()

logger = Data_collection(filtered_audio)



for i, sample in enumerate(filtered_audio):

    # processing the positive peaks
    
    # signal envelope
    top_envelope = top_envelope_detector.process(sample)
    
    # envelope slope direction
    top_slope = top_slope_detector.process(top_envelope)
    
    # if the slope starts to go down, a peak has just passed
    top_irq = top_interrupt.process(top_slope)
    
    
    top_tracker_peak, top_tracker_status = top_peak_tracker.process(sample, i, top_irq)

    logger.log_data("top", "envelope_signal", i, top_envelope)
    logger.log_data("top", "slope_polarity", i, top_slope)
    logger.log_data("top", "interrupt", i, top_irq)
    logger.log_data("top", "tracker", i, top_tracker_status)

    # processing the negative peaks
    sample = -sample # invert the sample so that the same methods can be reused as is for the bottom
    
    bottom_envelope = bottom_envelope_detector.process(sample)
    bottom_slope = bottom_slope_detector.process(bottom_envelope)
    bottom_irq = bottom_interrupt.process(bottom_slope)
    bottom_tracker_peak, bottom_tracker_status = bottom_peak_tracker.process(sample, i, bottom_irq)

    logger.log_data("bottom", "envelope_signal", i, bottom_envelope)
    logger.log_data("bottom", "slope_polarity", i, bottom_slope)
    logger.log_data("bottom", "interrupt", i, bottom_irq)
    logger.log_data("bottom", "tracker", i, bottom_tracker_status)

    # check for true peaks and wiggle the flip flops
    if top_tracker_peak == 0:
        period_doubler.set()
    elif bottom_tracker_peak == 0:
        period_doubler.reset()
    
    square_wave_output = period_doubler.process()
    logger.log_square_wave(i, square_wave_output)
    
    
logger.plot()

# multiply the clean audio with the square wave, every other period will be zero
result = clean_audio * logger.get_square_wave()

# filter the result to remove the harsh transitions
result = bandpass_filter(result, 2, 1, 2500, samplerate)

# write the output file
sf.write(output_file, result, samplerate)
    
