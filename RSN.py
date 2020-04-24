import numpy as np
import time
from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
import matplotlib.pyplot as plt


# Prepare simulation components
simLength=1024
model=models.Generic2dOscillator(a=np.array(0.0))
connectivity=connectivity.Connectivity.from_file("connectivity_76.zip")
coupling=coupling.Linear(a=np.array(6e-2))
integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([5e-4]))) # 2000 Hz
EEGmonitor = monitors.EEG(projection=projections.ProjectionSurfaceEEG.from_file("projection_eeg_65_surface_16k.npy"),
                           sensors=sensors.SensorsEEG.from_file("eeg_brainstorm_65.txt"),
                           region_mapping=region_mapping.RegionMapping.from_file("regionMapping_16k_76.txt"))
monitors = (monitors.Raw(), monitors.TemporalAverage(period=5.0), EEGmonitor)

# Run simulation
tic = time.time()
sim = simulator.Simulator(model, connectivity, coupling, integrator, monitors)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds." % (time.time() - tic,))

# Extract data: "output[a][b][:,0,:,0].T"
# where a=monitorIndex, b=(data:1,time:0) and [:,0,:,0].T arranges channel x timepoints.

raw_data = output[0][1][:,0,:,0].T
raw_time = output[0][0]
EEG_data = output[2][1][:,0,:,0].T
EEG_time = output[2][0]

# Plot raw time series
plt.figure(1)
plt.plot(raw_time, raw_data[:, 0, :, 0])
plt.title("Raw -- State variable 0")



from toolbox import timeseriesPlot, FFTplot, bandpassFIRfilter, CORR, PLV, PLI, AEC  # Home-made code
plt.style.use("pastelJescab01")  # seaborn | seaborn-whitegrid | ggplot | seaborn-paper | seaborn-pastel | pastelJescab01

# Plot EEG time series
ch_names = EEGmonitor.sensors.labels
timeseriesPlot(EEG_data, EEG_time, ch_names)

# Fourier Analysis plot
FFTplot(EEG_data, simLength)

# Band-pass filtering
# Famous window functions are: bartlett, hann, hamming, blackmann
# Last command (plot) if ON, the function plots each signal before and after filtering
filterSignals = bandpassFIRfilter(EEG_data, 8, 13, "hann", 2000, "ON")

# CONNECTIVITY MEASURES
## Correlation
correlation=CORR(EEG_data, ch_names, plot="ON")

## PLV
PLV=PLV(filterSignals)

## PLI
PLI=PLI(filterSignals)

## AEC
AEC=AEC(filterSignals, nEpochs=32)




x=np.reshape(plv,65*65)
y=np.reshape(pli,65*65)     # No correlacionan en absoluto. ¿Es posible?
np.corrcoef(x,y)




