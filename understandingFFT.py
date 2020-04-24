'# Understanding FFT'
from matplotlib.pyplot import *
import numpy

'''
# FFT transforma una señal del dominio del tiempo al dominio de las frecuencias.
# La FT es simétrica porque las frecuencias negativas y positivas configuran exactamente la misma onda. 
# El numero de frecuencias que podrás analizar con la FFT será la mitad del número de puntos que compongan la señal.
# El valor de esas frecuencias coincide con dividir range(int(len(signal)))/signalTime. [solo las positivas: simétrico]
# Con numpy.fft.fft calculas los vectores para cada frecuencia en complejo: obten los valores absolutos de cada vector.
# Obtendras las frecuencias a que corresponden esos módulos con numpy.fft.fftfreq(signalPoints, samplingRate), con esos 
## dos parámetros puede tener en cuenta la duración de la señal en tiempo para definir las frecuencias adecuadas.
'''

# 1a-. Simulemos una señal de 12 segundos con una frecuencia de 3Hz y un sampleo cada 0.01s
t = numpy.arange(0, 12, 0.01)
w = numpy.sin(2 * numpy.pi * 3 * t)
figure(1)
plot(t, w)
# 1b-. Si aplicamos el FFT directamente, no obtenemos la transformación que buscábamos
module_complex = numpy.fft.fft(w)
freq = numpy.fft.fftfreq(len(w), 0.01)
figure(2)
plot(freq, module_complex)

# 2-.Obteniendo los modulos de la fft tendremos un plot limpio. Aquí además se seleccionan solo valores positivos.
# Comprobamos el funcionamiento de fft.fftfreq obteniendo freq1 manualmente (rango frecuencias de la parte positiva).
module_abs = abs(module_complex[range(int(len(w)))])
freq1 = numpy.arange(len(w))
freq1 = freq1 / 12  # 12 = segundos de señal.
figure(3)
plot(freq1, module_abs)

###############################
# Tutorial from: https://pythontic.com/visualization/signals/fouriertransform_fft
# The Python example creates two sine waves and they are added together to create one signal. When the Fourier transform
# is applied to the resultant signal it provides the frequency components present in the sine wave.


# Python example - Fourier transform using numpy.fft method
import numpy as np
import matplotlib.pyplot as plotter

# How many time points are needed i,e., Sampling Frequency
samplingFrequency = 100;

# At what intervals time points are sampled
samplingInterval = 1 / samplingFrequency;

# Begin time period of the signals
beginTime = 0;

# End time period of the signals
endTime = 10;

# Frequency of the signals
signal1Frequency = 4;
signal2Frequency = 7;

# Time points
time = np.arange(beginTime, endTime, samplingInterval);

# Create two sine waves
amplitude1 = np.sin(2 * np.pi * signal1Frequency * time)
amplitude2 = np.sin(2 * np.pi * signal2Frequency * time)

# Create subplot
figure, axis = plotter.subplots(4, 1)
plotter.subplots_adjust(hspace=1)

# Time domain representation for sine wave 1
axis[0].set_title('Sine wave with a frequency of 4 Hz')
axis[0].plot(time, amplitude1)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

# Time domain representation for sine wave 2
axis[1].set_title('Sine wave with a frequency of 7 Hz')
axis[1].plot(time, amplitude2)
axis[1].set_xlabel('Time')
axis[1].set_ylabel('Amplitude')

# Add the sine waves
amplitude = amplitude1 + amplitude2

# Time domain representation of the resultant sine wave
axis[2].set_title('Sine wave with multiple frequencies')
axis[2].plot(time, amplitude)
axis[2].set_xlabel('Time')
axis[2].set_ylabel('Amplitude')

# Frequency domain representation
fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency

tpCount = len(amplitude)  # number of time points
values = np.arange(
    int(tpCount / 2))  # Numeric sequence from 1 to half time points. That's all freqs that can be assessed, why?
timePeriod = tpCount / samplingFrequency  # Time period dividing time points by sampling frequency. More Hz, less recording time.
frequencies = values / timePeriod  # You have the number of posible freqs to be assessed, but you dont have the exact freq:
# set those freqs dividing "values" by time period.

# Frequency domain representation
axis[3].set_title('Fourier transform depicting the frequency components')

axis[3].plot(frequencies, abs(fourierTransform))
axis[3].set_xlabel('Frequency')
axis[3].set_ylabel('Amplitude')

plotter.show()
