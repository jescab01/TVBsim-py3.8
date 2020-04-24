import numpy as np
import time
from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
import matplotlib.pyplot as plt

plt.style.use("pastelJescab01")  # seaborn | seaborn-whitegrid | ggplot | seaborn-paper | seaborn-pastel | pastelJescab01

tic = time.time()
sim = simulator.Simulator(
    model=models.Generic2dOscillator(a=np.array(0.0)),
    connectivity=connectivity.Connectivity.from_file("connectivity_76.zip"),
    coupling=coupling.Linear(a=np.array(6e-2)),
    integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([5e-4]))), # 2000 Hz
    monitors=(monitors.Raw(),
              monitors.TemporalAverage(period=5.0), # 200Hz
              monitors.EEG(projection=projections.ProjectionSurfaceEEG.from_file("projection_eeg_65_surface_16k.npy"),
                           sensors=sensors.SensorsEEG.from_file("eeg_brainstorm_65.txt"),
                           region_mapping=region_mapping.RegionMapping.from_file("regionMapping_16k_76.txt")))
)
sim.configure()

output = sim.run(simulation_length=1024)
print("simulation required %0.3f seconds." % (time.time() - tic,))

raw_data = output[0][1]
raw_time = output[0][0]
tavg_data = output[1][1]
tavg_time = output[1][0]
EEG_data = output[2][1]
EEG_data_norm = EEG_data/(np.max(EEG_data,0) - np.min(EEG_data,0))
EEG_time = output[2][0]

# Calculate PLV
from PLV import PLV
plv=PLV("alpha", EEG_data)


# Plot PLV
fig, ax = plt.subplots()
im = ax.imshow(plv)
fig.tight_layout()

# We want to show all ticks...
ax.set_xticks(np.arange(len(plv)))
ax.set_yticks(np.arange(len(plv)))
# ... and label them with the respective list entries
ax.set_xticklabels(ch_names)
ax.set_yticklabels(ch_names)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

ax.set_title("Phase Locking Values")
fig.tight_layout()
plt.show()




# Plot raw time series
plt.figure(1)
plt.plot(raw_time, raw_data[:, 0, :, 0])
plt.title("Raw -- State variable 0")

# Plot temporally averaged time series
plt.figure(2)
plt.plot(tavg_time, tavg_data[:, 0, :, 0])
plt.title("Temporal average")

# Plot EEG time series
ch_names=monitors.EEG(projection=projections.ProjectionSurfaceEEG.from_file("projection_eeg_65_surface_16k.npy"),
                           sensors=sensors.SensorsEEG.from_file("eeg_brainstorm_65.txt"),
                           region_mapping=region_mapping.RegionMapping.from_file("regionMapping_16k_76.txt"))
ch_names=ch_names.sensors.labels
from EEG_plotter import plotSignals
plotSignals(EEG_data,EEG_time,ch_names)


# Fourier Analysis
for i in range(len(raw_data[0,0,:,0])):
    fft = abs(np.fft.fft(raw_data[:, 0, 1, 0]))  # FFT for each channel signal
    fft = fft[range(int(len(raw_data) / 2))]  # Select just positive side of the symmetric FFT
    freqs = np.arange(len(raw_data) / 2)
    freqs = freqs / 1.024  # 1.024secs = 1024ms

    plt.figure(4)
    plt.title("FFT")
    plt.plot(freqs, fft)
    plt.xlabel("Frequency - (Hz)")
    plt.ylabel("Module")

plt.show()

# # Connectivity analysis
# import tvb.analyzers.node_coherence as corr_coeff
# from tvb.datatypes.time_series import TimeSeriesRegion
#
# tsr = TimeSeriesRegion(connectivity=sim.connectivity,
#                        data=tavg_data[:,0,:,0],
#                        sample_period=sim.monitors[0].period)
# tsr.configure()
#
# corrcoeff_analyser = corr_coeff.CorrelationCoefficient(time_series=tsr)
# corrcoeff_data = corrcoeff_analyser.evaluate()
# corrcoeff_data.configure()
# FC = corrcoeff_data.array_data[..., 0, 0]