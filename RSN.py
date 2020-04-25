import numpy as np
import time
from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
from toolbox import timeseriesPlot, FFTplot, bandpassFIRfilter, plotConversions, CORR, PLV, PLI, AEC  # Home-made code

# Prepare simulation components
simLength = 1024 # ms
model = models.Generic2dOscillator(a=np.array(0.0))
connectivity = connectivity.Connectivity.from_file("connectivity_76.zip")
coupling = coupling.Linear(a=np.array(6e-2))
integrator = integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([5e-4])))  # 2000 Hz
EEGmonitor = monitors.EEG(projection=projections.ProjectionSurfaceEEG.from_file("projection_eeg_65_surface_16k.npy"),
                          sensors=sensors.SensorsEEG.from_file("eeg_brainstorm_65.txt"),
                          region_mapping=region_mapping.RegionMapping.from_file("regionMapping_16k_76.txt"))
monitors = (monitors.Raw(), monitors.TemporalAverage(period=5.0), EEGmonitor)

# Run simulation
tic = time.time()
sim = simulator.Simulator(model=model, connectivity=connectivity, coupling=coupling, integrator=integrator,
                          monitors=monitors)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds." % (time.time() - tic,))

# Extract data: "output[a][b][:,0,:,0].T"
# where a=monitorIndex, b=(data:1,time:0) and [:,0,:,0].T arranges channel x timepoints.
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0]
EEG_data = output[2][1][:, 0, :, 0].T
EEG_time = output[2][0]
ch_names = EEGmonitor.sensors.labels

# Plot raw time series
timeseriesPlot(EEG_data, EEG_time, ch_names)

# Fourier Analysis plot
FFTplot(EEG_data, simLength, ch_names)

# Band-pass filtering
# Famous window functions are: bartlett, hann, hamming, blackmann. plot="ON" plots 10 channels before and after filter.
filterSignals = bandpassFIRfilter(EEG_data, 8, 13, "hann", 2000, EEG_time, plot="ON")

# Check conversions (i.e. filtering, inst. phase, inst. amplitude envelope) for some signals
plotConversions(3, EEG_data, EEG_time, filterSignals, ch_names)

# CONNECTIVITY MEASURES
## Correlation
correlation = CORR(EEG_data, ch_names, plot="ON")

## PLV
PLV = PLV(filterSignals, ch_names, plot="ON")

## PLI
PLI = PLI(filterSignals, ch_names, plot="ON")

## AEC
AEC = AEC(filterSignals, nEpochs=32, ch_names=ch_names, plot="ON")

# x=np.reshape(plv,65*65)
# y=np.reshape(pli,65*65)     # No correlacionan en absoluto. ¿Es posible?
# np.corrcoef(x,y)
