import numpy as np
import time
from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
from toolbox import timeseriesPlot, FFTplot, bandpassFIRfilter, plotConversions, CORR, PLV, PLI, AEC  # Home-made code

# Prepare simulation components
simLength = 1024 # ms
m = models.ReducedWongWang()
#model = models.Generic2dOscillator(a=np.array(0.0))
conn = connectivity.Connectivity.from_file("connectivity_66.zip")
coup = coupling.Linear(a=np.array(6e-2))
int = integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([5e-4])))  # 2000 Hz
# EEGmonitor = monitors.EEG(projection=projections.ProjectionSurfaceEEG.from_file("projection_eeg_65_surface_16k.npy"),
#                           sensors=sensors.SensorsEEG.from_file("eeg_brainstorm_65.txt"),
#                           region_mapping=region_mapping.RegionMapping.from_file("regionMapping_16k_76.txt"))
RAWmonitor = monitors.Raw()
mon = (RAWmonitor, monitors.TemporalAverage(period=5.0))

# Run simulation
tic = time.time()
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds." % (time.time() - tic,))

# Extract data: "output[a][b][:,0,:,0].T"
# where a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
raw_data = output[0][1][200:, 0, :, 0].T
raw_time = output[0][0][200:]
regionLabels = conn.region_labels
# EEG_data = output[2][1][:, 0, :, 0].T
# EEG_time = output[2][0]
# ch_names = EEGmonitor.sensors.labels

# Plot raw time series
timeseriesPlot(raw_data, raw_time, regionLabels)

# Fourier Analysis plot
FFTplot(raw_data, simLength, regionLabels)

# Band-pass filtering
# Famous window functions are: bartlett, hann, hamming, blackmann. plot="ON" plots 10 channels before and after filter.
filterSignals = bandpassFIRfilter(raw_data, 8, 13, "hann", 2000, raw_time, plot="OFF")

# Check conversions (i.e. filtering, inst. phase, inst. amplitude envelope) for some signals
plotConversions(2, raw_data, raw_time, filterSignals, regionLabels)

# CONNECTIVITY MEASURES
## Correlation
correlation = CORR(raw_data, regionLabels, plot="ON")

## PLV
PLV = PLV(filterSignals, regionLabels, plot="ON")

## PLI
PLI = PLI(filterSignals, regionLabels, plot="ON")

## AEC
AEC = AEC(filterSignals, nEpochs=32, regionLabels=regionLabels, plot="ON")

# x=np.reshape(plv,65*65)
# y=np.reshape(pli,65*65)     # No correlacionan en absoluto. ¿Es posible?
# np.corrcoef(x,y)
