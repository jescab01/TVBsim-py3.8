# This is going to be a script with useful functions I will be using frequently.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, hilbert
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio


def timeseriesPlot(signals, timepoints, ch_names):
    fig = go.Figure(layout=dict(xaxis=dict(title='time'), yaxis=dict(title='Voltage')))
    for ch in range(len(signals)):
        fig.add_scatter(x=timepoints, y=signals[ch], name=ch_names[ch])
    pio.write_html(fig, file="figures/TimeSeries.html", auto_open=True)


def FFTplot(signals, simLength, ch_names):
    fig = go.Figure(layout=dict(xaxis=dict(title='Frequency'), yaxis=dict(title='Module')))
    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[0]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / (simLength/1000)  # simLength (ms) / 1000 -> segs

        fig.add_scatter(x=freqs, y=fft, name=ch_names[i])

    pio.write_html(fig, file="figures/FFT.html", auto_open=True)


def bandpassFIRfilter(signals, lowcut, highcut, windowtype, samplingRate, times, plot="OFF"):
    """
     Truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
     Some famous windows are: bartlett, hann, hamming and blackmann.
     http://www.labbookpages.co.uk/audio/firWindowing.html
     https://en.wikipedia.org/wiki/Window_function
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    """
    order = int(len(signals[0, :]) / 3)
    firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass", fs=samplingRate)
    filterSignals = filtfilt(b=firCoeffs, a=[1.0], x=signals, padlen=3 * order) # a=[1.0] as it is FIR filter (not IIR).

    if plot == "ON":
        plt.plot(range(len(firCoeffs)), firCoeffs) # Plot filter shape
        plt.title("FIR filter shape w/ %s windowing" % windowtype)
        for i in range(1, 10):
            plt.figure(i+1)
            plt.xlabel("time (ms)")
            plt.plot(times, signals[i], label="Raw signal")
            plt.plot(times, filterSignals[i], label="Filtered Signal")
        plt.show()
        plt.savefig("figures/filterSample%s" % str(i))
    return filterSignals


def plotConversions(n_signals, signals, timepoints, filterSignals, ch_names):

    phase = np.ndarray((len(filterSignals), len(filterSignals[0])))
    amplitude_envelope = np.ndarray((len(filterSignals), len(filterSignals[0])))

    for channel in range(n_signals):
        # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
        analyticalSignal = hilbert(filterSignals[channel])
        # Save instantaneous signal phase by channel
        phase[channel] = np.unwrap(np.angle(analyticalSignal))
        # Save absolute value of analytic signal
        amplitude_envelope[channel] = np.abs(analyticalSignal)

        for i in range(len(phase[channel])):
            if phase[channel, i] > np.pi:
                phase[channel, i:] = phase[channel, i:] - 2 * np.pi
        phase[channel]=phase[channel]*1000
        fig = go.Figure(layout=dict(xaxis=dict(title='Time (ms)'), yaxis=dict(title='Amplitude')))
        fig.add_scatter(x=timepoints, y=signals[channel], name="Raw signal")
        fig.add_scatter(x=timepoints, y=filterSignals[channel], name="Filtered signal")
        fig.add_scatter(x=timepoints, y=phase[channel], name="Instantaneous phase")
        fig.add_scatter(x=timepoints, y=amplitude_envelope[channel], name="Amplitude envelope")
        fig.update_layout(title="%s channel conversions" % ch_names[channel])
        pio.write_html(fig, file="figures/%s_channel_conversions.html" % ch_names[channel], auto_open=True)



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
        fig = go.Figure(data=go.Heatmap(z=CORR, x=ch_names, y=ch_names, colorscale='Viridis'))
        fig.update_layout(title='Correlation')
        pio.write_html(fig, file="figures/CORR.html", auto_open=True)

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
        fig = go.Figure(data=go.Heatmap(z=PLV, x=ch_names, y=ch_names, colorscale='Viridis'))
        fig.update_layout(title='Phase Locking Value')
        pio.write_html(fig, file="figures/PLV.html", auto_open=True)

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
            PLI[channel1, channel2] = np.absolute(sum(np.sign(phaseDifference))/len(phaseDifference))



    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=PLI, x=ch_names, y=ch_names, colorscale='Viridis'))
        fig.update_layout(title='Phase Lag Index')
        pio.write_html(fig, file="figures/PLI.html", auto_open=True)

    return PLI

def AEC(filterSignals, nEpochs, ch_names, plot="OFF"):
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
        fig = go.Figure(data=go.Heatmap(z=AEC, x=ch_names, y=ch_names, colorscale='Viridis'))
        fig.update_layout(title='Amplitude Envelope Correlation')
        pio.write_html(fig, file="figures/AEC.html", auto_open=True)

    return AEC

