import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plotSignals(data, times, ch_names):
    # Load the EEG data
    n_samples, n_rows = len(data[:, 0, 0, 0]), len(data[0, 0, :, 0])
    # with cbook.get_sample_data('eeg.dat') as eegfile:
    #    data = np.fromfile(eegfile, dtype=float).reshape((n_samples, n_rows))

    # Plot the EEG
    ticklocs = []
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    fig, ax2 = plt.subplots()
    ax2.set_ylim(y0, y1)
    ax2.set_xlim(0, times[-1])
    ax2.set_xticks(np.arange(0, times[-1], 100))

    segs = []
    for i in range(n_rows):
        segs.append(np.column_stack([times, data[:, 0, i, 0] + i * dr]))
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs)
    ax2.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    ax2.set_yticks(ticklocs)
    ax2.set_yticklabels(ch_names)
    ax2.set_xlabel('Time (s)')


