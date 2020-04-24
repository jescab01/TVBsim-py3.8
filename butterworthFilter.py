from scipy.signal import butter, lfilter

"""
First described in 1930 by the British engineer and physicist Stephen Butterworth 
in his paper entitled "On the Theory of Filter Amplifiers".
See https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass.html 
"""

# Si existe alguna forma de definir el order en función de la señal, defininr aquí el procedimiento

def butterworthFilter(signal, lowcut, highcut, samplingRate, order=4):
   nyq = 0.5 * samplingRate
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band') # Define filter shape
   filteredSignal = lfilter(b, a, signal) # Apply filter to signal
   return filteredSignal
