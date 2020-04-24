import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt


# Filtrando con butterworth que tengo
order=4 #¿?¿?
nyq = 0.5 * samplingRate
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='band')  # Define filter shape
filsig = lfilter(b, a, signal)  # Apply filter to signal

plotsigs(filsig)
fftplot(signal)
fftplot(filsig)


# butterworth que tengo mas el filtfilt
order=4 #¿?¿?
nyq = 0.5 * samplingRate
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='band')
filsig = filtfilt(b,a,signal)  # Apply filter to signal

plotsigs(filsig)
fftplot(signal)
fftplot(filsig)


# Filtrando con mne.filter.data_filter

filsig=mne.filter.filter_data(signal, 2000, 8, 13)


# Filtrando con scipy.signal.irrfilter

#####################################
# First things first: Create a filter
from scipy.signal import firwin, lfilter, filtfilt
import matplotlib.pyplot as plt

signalpoints=2048
samplingRate=2000
order=int(signalpoints/3)
lowcut=8
highcut=13
signal=EEG_data[:,0,1,0]
# truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
# go to: http://www.labbookpages.co.uk/audio/firWindowing.html
# go to: https://en.wikipedia.org/wiki/Window_function
# go to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
# famous windows are: bartlett, hann, hamming, blackmann
window = "hamming"

a=firwin(numtaps=order+1, cutoff=[lowcut,highcut], window="hann", pass_zero="bandpass", fs=2000)
plt.plot(range(len(a)), a)

filsig=lfilter(b=a,a=[1.0],x=signal)
filfil=filtfilt(b=a,a=[1.0],x=signal, padlen=3*order)

plt.plot(range(len(signal)), signal)
plt.plot(range(len(signal)), filsig)
plt.plot(range(len(signal)), filfil)








