# This is going to be a script with useful functions I will be using frequently.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, hilbert
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import math
import time


def timeseriesPlot(signals, timepoints, regionLabels):
    fig = go.Figure(layout=dict(xaxis=dict(title='time'), yaxis=dict(title='Voltage')))
    for ch in range(len(signals)):
        fig.add_scatter(x=timepoints, y=signals[ch], name=regionLabels[ch])
    pio.write_html(fig, file="figures/TimeSeries.html", auto_open=True)

def epochingTool(signals, epoch_length, samplingFreq):
    """
    Epoch length in seconds; sampling frequency in Hz
    """
    tic = time.time()
    nEpochs=math.trunc(len(signals[0])/(epoch_length*samplingFreq))
    epochedSignals=list()
    print("Epoching signals", end="")
    for channel in range(len(signals)):
        epochedSignals.append(np.array_split(signals[channel], nEpochs))

    print(" - %0.3f seconds.\n" % (time.time() - tic,))
    return epochedSignals

def FFTplot(signals, simLength, regionLabels):
    fig = go.Figure(layout=dict(xaxis=dict(title='Frequency'), yaxis=dict(title='Module')))
    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[0]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / (simLength/1000)  # simLength (ms) / 1000 -> segs

        fig.add_scatter(x=freqs, y=fft, name=regionLabels[i])

    pio.write_html(fig, file="figures/FFT.html", auto_open=True)


def bandpassFIRfilter(signals, lowcut, highcut, windowtype, samplingRate, times=None, plot="OFF"):
    """
     Truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
     Some famous windows are: bartlett, hann, hamming and blackmann.
     Two processes depending on input: epoched signals or entire signals
     http://www.labbookpages.co.uk/audio/firWindowing.html
     https://en.wikipedia.org/wiki/Window_function
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    """
    tic = time.time()
    try:
        signals[0][0][0]
        order = int(len(signals[0][0]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass",
                           fs=samplingRate)
        efsignals=signals
        print("Band pass filtering epoched signals: %i-%iHz " % (lowcut, highcut), end="")
        for channel in range(len(signals)):
            print(".", end="")
            for epoch in range(len(signals[0])):
                efsignals[channel][epoch] = filtfilt(b=firCoeffs, a=[1.0], x=signals[channel][epoch],padlen=int(2.5 * order))
                # a=[1.0] as it is FIR filter (not IIR).
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return efsignals

    except IndexError:
        order = int(len(signals[0, :]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass", fs=samplingRate)
        filterSignals = filtfilt(b=firCoeffs, a=[1.0], x=signals, padlen=int(2.5 * order)) # a=[1.0] as it is FIR filter (not IIR).
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


def plotConversions(n_signals, signals, timepoints, filterSignals, regionLabels):

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


        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_scatter(x=timepoints, y=signals[channel], name="Raw signal")
        fig.add_scatter(x=timepoints, y=filterSignals[channel], name="Filtered signal")
        fig.add_scatter(x=timepoints, y=phase[channel], name="Instantaneous phase", secondary_y=True)
        fig.add_scatter(x=timepoints, y=amplitude_envelope[channel], name="Amplitude envelope")

        fig.update_layout(title="%s channel conversions" % regionLabels[channel])
        fig.update_xaxes(title_text="Time (ms)")

        fig.update_yaxes(title_text="Amplitude", range=[-max(signals[channel]),max(signals[channel])], secondary_y=False)
        fig.update_yaxes(title_text="Phase", tickvals=[-3.14, 0, 3.14], range=[-15, 15], secondary_y=True, gridcolor='mintcream')

        pio.write_html(fig, file="figures/%s_channel_conversions.html" % regionLabels[channel], auto_open=True)

def CORR(signals, regionLabels, plot="OFF"):
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
        fig = go.Figure(data=go.Heatmap(z=CORR, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Correlation')
        pio.write_html(fig, file="figures/CORR.html", auto_open=True)

    return CORR

def PLV(filterSignals, regionLabels=None, plot="OFF"):
    tic = time.time()
    try:
        filterSignals[0][0][0]
        # Create an array (channels x time) to save instantaneous phase data.
        PLV = np.ndarray(((len(filterSignals)), len(filterSignals)))
        phase = filterSignals
        # For each signal: hilbert transform to extract analytical signal -> extract instantaneous phase
        for channel in range(len(filterSignals)):
            for epoch in range(len(filterSignals[0])):
                # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
                analyticalSignal = hilbert(filterSignals[channel][epoch])
                # Save instantaneous signal phase by channel
                phase[channel][epoch] = np.unwrap(np.angle(analyticalSignal))

        print("Calculating PLV", end="")
        for channel1 in range(len(filterSignals)):
            print(".", end="")
            for channel2 in range(len(filterSignals)):
                plv_values = list()
                for epoch in range(len(filterSignals[0])):
                    phaseDifference = phase[channel1][epoch] - phase[channel2][epoch]
                    pdCos = np.cos(phaseDifference)  # Phase difference cosine
                    pdSin = np.sin(phaseDifference)  # Phase difference sine
                    value = np.sqrt(sum(pdCos) ** 2 + sum(pdSin) ** 2) / len(filterSignals[0][0])
                    plv_values.append(value)
                    PLV[channel1, channel2] = np.average(plv_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Locking Value')
            pio.write_html(fig, file="figures/PLV.html", auto_open=True)
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLV

    except IndexError:
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
            fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Locking Value')
            pio.write_html(fig, file="figures/PLV.html", auto_open=True)
        return PLV


def PLI(filterSignals, regionLabels=None, plot="OFF"):
    tic = time.time()
    try:
        filterSignals[0][0][0]

        PLI = np.ndarray(((len(filterSignals)), len(filterSignals)))
        # Create an array (channels x time) to save instantaneous phase data.
        phase = filterSignals
        # For each signal: hilbert transform to extract analytical signal -> extract instantaneous phase
        for channel in range(len(filterSignals)):
            for epoch in range(len(filterSignals[0])):
                # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
                analyticalSignal = hilbert(filterSignals[channel][epoch])
                # Save instantaneous signal phase by channel
                phase[channel][epoch] = np.unwrap(np.angle(analyticalSignal))

        print("Calculating PLI", end="")
        for channel1 in range(len(filterSignals)):
            print(".", end="")
            for channel2 in range(len(filterSignals)):
                pli_values = list()
                for epoch in range(len(filterSignals[0])):
                    phaseDifference = phase[channel1][epoch] - phase[channel2][epoch]
                    pli_values.append(np.absolute(sum(np.sign(phaseDifference)) / len(phaseDifference)))
                    PLI[channel1, channel2] = np.average(pli_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLI, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Lag Index')
            pio.write_html(fig, file="figures/PLI.html", auto_open=True)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLI

    except IndexError:
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
            fig = go.Figure(data=go.Heatmap(z=PLI, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Lag Index')
            pio.write_html(fig, file="figures/PLI.html", auto_open=True)

        return PLI

def AEC(filterSignals, regionLabels=None, plot="OFF"):
    """
    nEpochs should be a divisor of number of signal points.
    """
    tic = time.time()
    try:
        filterSignals[0][0][0]
        AEC = np.ndarray(((len(filterSignals)), len(filterSignals))) # averaged AECs per channel x channel
        amplitude_envelope = filterSignals
        normalizedAE = filterSignals # array for normalized Amplitude Envelope
        # For each signal: filter it for specific frequency band -> extract analytical signal -> extract instantaneous phase
        for channel in range(len(filterSignals)):
            for epoch in range(len(filterSignals[0])):
                # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
                analyticalSignal = hilbert(filterSignals[channel][epoch])
                # Save absolute value of analytic signal
                amplitude_envelope[channel][epoch] = np.abs(analyticalSignal)

                # Normalize amplitude envelope preparing for AEC
                mean = np.mean(amplitude_envelope[channel][epoch])
                std = np.std(amplitude_envelope[channel][epoch])
                normalizedAE[channel][epoch] = (amplitude_envelope[channel][epoch] - mean) / std
        print("Calculating AEC", end="")
        for channel1 in range(len(filterSignals)):
            print(".", end="")
            for channel2 in range(len(filterSignals)):
                values_aec = list()  # AEC per epoch and channel x channel
                for epoch in range(len(filterSignals[0])): # CORR between channels by epoch
                    values_aec.append(sum(normalizedAE[channel1][epoch]*normalizedAE[channel2][epoch])/len(normalizedAE[channel1][epoch]))
                AEC[channel1, channel2] = np.average(values_aec)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=AEC, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Amplitude Envelope Correlation')
            pio.write_html(fig, file="figures/AEC.html", auto_open=True)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return AEC


    except IndexError:
        print("Signals must be epoched before calculating AEC. Use epochingTool().")

