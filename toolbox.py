# This is going to be a script with useful functions I will be using frequently.
# This are the functions:
#   -

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import firwin, filtfilt, hilbert

def timeseriesPlot(signals, timepoints, ch_names):
    # Load the EEG data
    n_rows, n_samples = len(signals[:, 0]), len(signals[0])

    # Plot the EEG
    ticklocs = []
    dmin = signals.min()
    dmax = signals.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    fig, ax2 = plt.subplots()
    ax2.set_ylim(y0, y1)
    ax2.set_xlim(0, times[-1])
    ax2.set_xticks(np.arange(0, times[-1], 100))

    segs = []
    for i in range(n_rows):
        segs.append(np.column_stack([times, signals[i, :] + i * dr]))
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs)
    ax2.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    ax2.set_yticks(ticklocs)
    ax2.set_yticklabels(ch_names)
    ax2.set_xlabel('Time (s)')


def FFTplot(signals, simulation_length):
    for i in range(len(signals[:, 0])):
        fft = abs(np.fft.fft(signals[i, :]))  # FFT for each channel signal
        fft = fft[range(int(len(signals) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / simLength  # (e.g. 1.024secs = 1024ms)

        plt.plot(freqs, fft)
        plt.title("FFT")
        plt.xlabel("Frequency - (Hz)")
        plt.ylabel("Module")


def bandpassFIRfilter(signals, lowcut, highcut, windowtype, samplingRate, plot="OFF"):
    """
     Truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
     Some famous windows are: bartlett, hann, hamming and blackmann.
     http://www.labbookpages.co.uk/audio/firWindowing.html
     https://en.wikipedia.org/wiki/Window_function
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    """
    order = int(len(signals[0,:]) / 3)
    firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass", fs=samplingRate)
    filterSignals = filtfilt(b=filterCoeffs, a=[1.0], x=signal, padlen=3 * order) # a=[1.0] as it is FIR filter (not IIR).

    if plot == "ON":
        plt.plot(range(len(firCoeffs)), firCoeffs) # Plot filter shape
        plt.title("FIR filter shape w/ %s windowing" % windowtype)
        for i in range(len(signals)):
            plt.figure(i+1)
            plt.plot(range(len(signals[0])), signals[i])
            plt.plot(range(len(signals[0])), filterSignals[i])

    return filterSignals

def CORR(signals, ch_names, plot="OFF"):
    """
    To compute correlation between signals you need to standarize signal values and then to sum up intersignal products
    divided by signal length.
    """

    normalSignals = np.ndarray((len(signals), len(signals[0])))
    for channel in range(len(signals)):
        mean = np.mean(signals[channel,:])
        std = np.std(signals[channel,:])
        normalSignals[channel] = (signals[channel,:] - mean) / std

    CORR = np.ndarray((len(normalSignals), len(normalSignals)))
    for channel1 in range(len(normalSignals)):
        for channel2 in range(len(normalSignals)):
            CORR[channel1][channel2] = sum(normalSignals[channel1] * normalSignals[channel2]) / len(normalSignals[0])

    if plot == "ON":
        # Plot CORR
        fig, ax = plt.subplots()
        im = ax.imshow(CORR)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(CORR)))
        ax.set_yticks(np.arange(len(CORR)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(ch_names)
        ax.set_yticklabels(ch_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Pearson Correlation")
        fig.tight_layout()

    return CORR

def PLV(filterSignals, ch_names, plot="OFF"):

    # Create an array (channels x time) to save instantaneous phase data.
    PLV = np.ndarray(((len(filterSignals)), len(filterSignals)))
    phase = np.ndarray((len(filterSignals), len(filterSignals[0])))

    # For each signal: hilbert transform to extract analytical signal -> extract instantaneous phase
    for channel in range(len(filterSignals)):
        # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
        analyticalSignal = hilbert(filterSignals[channel])
        # Save instantaneous signal phase by channel
        phase[channel] = np.unwrap(np.angle(analyticalSignal))

    for channel1 in range(len(filterSignals[:, 0])):
        for channel2 in range(len(filterSignals[:, 0])):
            phaseDifference = phase[channel1] - phase[channel2]
            pdCos = np.cos(phaseDifference)  # Phase difference cosine
            pdSin = np.sin(phaseDifference)  # Phase difference sine
            PLV[channel1, channel2] = np.sqrt(sum(pdCos) ** 2 + sum(pdSin) ** 2) / len(filterSignals[0])

    if plot == "ON":
        # Plot PLV
        fig, ax = plt.subplots()
        im = ax.imshow(PLV)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(PLV)))
        ax.set_yticks(np.arange(len(PLV)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(ch_names)
        ax.set_yticklabels(ch_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Phase Locking Value")
        fig.tight_layout()

    return PLV


def PLI(filterSignals, ch_names, plot="OFF"):

    PLI = np.ndarray(((len(filterSignals)), len(filterSignals)))
    # Create an array (channels x time) to save instantaneous phase data.
    phase = np.ndarray((len(filterSignals), len(filterSignals[0])))
    # For each signal: hilbert transform to extract analytical signal -> extract instantaneous phase
    for channel in range(len(filterSignals)):
        # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
        analyticalSignal = hilbert(filterSignals[channel])
        # Save instantaneous signal phase by channel
        phase[channel] = np.unwrap(np.angle(analyticalSignal))

    for channel1 in range(len(filterSignals)):
        for channel2 in range(len(filterSignals)):
            phaseDifference = phase[channel1] - phase[channel2]
            pdCos = np.cos(phaseDifference)  # Phase difference cosine
            pdSin = np.sin(phaseDifference)  # Phase difference sine
            PLI[channel1, channel2] = np.absolute(sum(np.sign(phaseDifference))/len(phaseDifference))

    PLI[PLI==0]=1

    if plot == "ON":
        # Plot PLV
        fig, ax = plt.subplots()
        im = ax.imshow(PLI)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(PLI)))
        ax.set_yticks(np.arange(len(PLI)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(ch_names)
        ax.set_yticklabels(ch_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Phase Lag Index")
        fig.tight_layout()

    return PLI

def AEC(filterSignals, nEpochs):
    """
    nEpochs should be a divisor of number of signal points.
    """
    amplitude_envelope = np.ndarray((len(filterSignals), len(filterSignals[0])))
    normalizedAE = np.ndarray((len(filterSignals), len(filterSignals[0]))) # array for normalized Amplitude Envelope
    ampEpochs = np.ndarray((len(filterSignals), nEpochs, int(len(filterSignals[0])/nEpochs)))
    # For each signal: filter it for specific frequency band -> extract analytical signal -> extract instantaneous phase
    for channel in range(len(filterSignals)):
        # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
        analyticalSignal = hilbert(filterSignals[channel])
        # Save absolute value of analytic signal
        amplitude_envelope[channel] = np.abs(analyticalSignal)

        # Normalize si amplitude envelope preparing for AEC
        mean = np.mean(amplitude_envelope[channel])
        std = np.std(amplitude_envelope[channel])
        normalizedAE[channel] = (amplitude_envelope[channel] - mean) / std

        # Epoching for AEC
        ampEpochs[channel] = np.asarray(np.split(normalizedAE[channel], nEpochs))

    AECs = np.ndarray(((len(filterSignals)), len(filterSignals), nEpochs)) # AEC per epoch and channel x channel
    AEC = np.ndarray(((len(filterSignals)), len(filterSignals))) # averaged AECs per channel x channel

    for channel1 in range(len(filterSignals)):
        for channel2 in range(len(filterSignals)):
            for epoch in range(nEpochs): # CORR between channels by epoch
                AECs[channel1, channel2][epoch] = sum(ampEpochs[channel1][epoch]*ampEpochs[channel2][epoch])/len(ampEpochs[channel1][epoch])
            AEC[channel1, channel2]=np.average(AECs[channel1,channel2])

    if plot == "ON":
        # Plot AEC
        fig, ax = plt.subplots()
        im = ax.imshow(AEC)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(AEC)))
        ax.set_yticks(np.arange(len(AEC)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(ch_names)
        ax.set_yticklabels(ch_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Amplitude Envelope Correlation")
        fig.tight_layout()

    return AEC

