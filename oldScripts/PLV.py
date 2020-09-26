import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from oldScripts.butterworthFilter import butterworthFilter

data, lowcut, highcut, samplingRate = EEG_data, 4,8,2000

def PLV(data, lowcut, highcut, samplingRate):
    fBands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70)]  # Pair frequencies (low-high) by band.
    nBands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]  # Frequency band names matching indexes in previous array.
    # Print initial message
    if (lowcut, highcut) in fBands:
        print("Calculating PLV for %s band" % nBands[fBands == (lowcut, highcut)])
    else:
        print("Calculating PLV with %s - %s Hz band." % (str(lowcut), str(highcut)))
        print("Not a defined frequency band though: delta[1-4], theta[4-8], alpha[8-13], beta[13-30], gamma[30-70]")

    # Create an array (channels x time) to save instantaneous phase data.
    phase = np.ndarray((len(data[0, 0, :, 0]), len(data[:, 0, 0, 0])))
    amplitude_envelope = np.ndarray((len(data[0, 0, :, 0]), len(data[:, 0, 0, 0])))
    nAE = np.ndarray((len(data[0, 0, :, 0]), len(data[:, 0, 0, 0])))
    nEpochs = 32 # must be a divisor of signal points
    ampEpochs = np.ndarray((len(data[0, 0, :, 0]), nEpochs, int(len(data[:, 0, 0, 0])/nEpochs)))
    # For each signal: filter it for specific frequency band -> extract analytical signal -> extract instantaneous phase
    for channel in range(len(data[0, 0, :, 0])):
        # Call butterworth function -homemade- with its arguments: data, lowcut, highcut, samplingRate, order.
        filteredSignal=butterworthFilter(data[:,0,channel,0],lowcut,highcut,samplingRate,2)
        # Hilbert transform to extract analytical signal (i.e. with instantaneous phase and amplitude). Scipy function.
        analyticalSignal = hilbert(filteredSignal)
        # Save instantaneous signal phase by channel
        phase[channel] = np.unwrap(np.angle(analyticalSignal))
        # Save absolute value of analytic signal
        amplitude_envelope[channel] = np.abs(analyticalSignal)

        # Normalize si amplitude envelope preparing for AEC
        mean = np.mean(amplitude_envelope[channel])
        std = np.std(amplitude_envelope[channel])
        nAE[channel] = (amplitude_envelope[channel] - mean) / std

        # Epoching for AEC
        ampEpochs[channel] = np.asarray(np.split(nAE[channel],nEpochs))



    PLV = np.ndarray(((len(data[0, 0, :, 0])), len(data[0, 0, :, 0])))
    PLI = np.ndarray(((len(data[0, 0, :, 0])), len(data[0, 0, :, 0])))

    AECs = np.ndarray(((len(data[0, 0, :, 0])), len(data[0, 0, :, 0]), nEpochs))
    AEC = np.ndarray(((len(data[0, 0, :, 0])), len(data[0, 0, :, 0])))

    for channel1 in range(len(data[0, 0, :, 0])):
        for channel2 in range(len(data[0, 0, :, 0])):
            phaseDifference = phase[channel1] - phase[channel2]
            pdCos = np.cos(phaseDifference)  # Phase difference cosine
            pdSin = np.sin(phaseDifference)  # Phase difference sine
            PLV[channel1, channel2] = np.sqrt(sum(pdCos) ** 2 + sum(pdSin) ** 2) / len(data[:, 0, 0, 0])

            PLI[channel1, channel2] = np.absolute(sum(np.sign(phaseDifference))/len(phaseDifference))

            for epoch in range(nEpochs): # CORR between channels by epoch
                AECs[channel1, channel2][epoch] = sum(ampEpochs[channel1][epoch]*ampEpochs[channel2][epoch])/len(ampEpochs[channel1][epoch])

            AEC[channel1, channel2]=np.average(AECs[channel1,channel2])


    PLI[PLI==0]=1

    return PLV, PLI



def plotPLV(plv, ch_names):
    # Plot PLV
    fig, ax = plt.subplots()
    im = ax.imshow(plv)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(plv)))
    ax.set_yticks(np.arange(len(plv)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ch_names)
    ax.set_yticklabels(ch_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right",
             rotation_mode="anchor")

    ax.set_title("Phase Locking Values")
    fig.tight_layout()
