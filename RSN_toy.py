import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool,bandpassFIRfilter,plotConversions, timeseriesPlot, FFTplot  # Home-made code
import mne.filter

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
# subjectid = "FC_subjx"


# Prepare simulation parameters
simLength = 1000 # ms
samplingFreq = 1000 #Hz

jrm = models.Generic2dOscillator()
# phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max-jrm.p_min) * 0.5) ** 2 / 2.
# sigma = np.zeros(6)
# sigma[3] = phi_n_scaling

speed = 4  # mm/ms
coup = coupling.Linear(a=np.array([0.5]))
int = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([0.0001]))) # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
mon = (monitors.Raw(), monitors.TemporalAverage(period=1.0))


#define STRUCTURE
conn = connectivity.Connectivity.from_file("toy.zip")
# conn.weights[1,0]=0.5
# conn.weights[2,0]=-1
# conn.weights[3,0]=0.04

# # # conn.weights[1,2]=4
# conn.weights[0,3]=0.1
# conn.weights = conn.scaled_weights(mode="tract") # Scales by matrix's max value

# conn.tract_lengths[0,1]=0

# #STIMULATE
# w=np.zeros(len(conn.weights[0]))
# w[[0]]=0.5
#
# eqn_t = equations.PulseTrain()
# eqn_t.parameters['onset'] = 200.0
# eqn_t.parameters['T'] = 300.0
# eqn_t.parameters['tau'] = 5.0
#
# stimulus = patterns.StimuliRegion(temporal=eqn_t,connectivity=conn,weight=w)

# #Configure space and time
# stimulus.configure_space()
# stimulus.configure_time(np.arange(0., 3e3, 2**-4))
#
# #And take a look
# plot_pattern(stimulus)


# Run simulation
tic = time.time()
sim = simulator.Simulator(model=jrm, connectivity=conn, coupling=coup, conduction_speed=speed,  integrator=int, monitors=mon)#, stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=1000)
print("simulation required %0.3f seconds.\n" % (time.time() - tic,))

# Extract data
raw_data = output[0][1][100:, 0, :, 0].T
raw_time = output[0][0][100:]
regionLabels = conn.region_labels


# Plot raw time series
timeseriesPlot(raw_data, raw_time, regionLabels)

# del coup, int, jrm, models, mon, sim, subjectid, tic


# Fourier Analysis plot
FFTplot(raw_data, simLength, regionLabels)

# # Epoch timeseries into x seconds windows epochingTool(data, windowlength, samplingFrequency)
# epochedData = epochingTool(raw_data, 4, samplingFreq)
# timeseriesPlot(epochedData[0],np.array(range(len(epochedData[0][0]))),regionLabels)
#
#
#
# bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
# filterSignals=np.ndarray((len(bands[0]), len(epochedData), len(epochedData[0]), len(epochedData[0][0])))
# for b in range(len(bands[0])):
#     (lowcut, highcut) = bands[1][b]
#
#     # Band-pass filtering
#     # Famous window functions are: bartlett, hann, hamming, blackmann.
#     filterSignals[b]=mne.filter.filter_data(epochedData,1000,lowcut,highcut)
#
#     # Check conversions (i.e. filtering, inst. phase, inst. amplitude envelope) for some signals
#     plotConversions(2, epochedData, raw_time, filterSignals[b], regionLabels, n_epochs=1)
#
# aec1=mne.connectivity.envelope_correlation(filterSignals[2])

# # Copy structural connetivity weights in FC folder
# weights=conn.weights
# fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\weights.txt"
# np.savetxt(fname, weights)

# w=conn.weights
# aec=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\3-alfacorramp.txt")
#
#
# t3 = np.zeros(shape=(2, 2145))
# t3[0, :] = aec[np.triu_indices(66, 1)]
# t3[1, :] = w[np.triu_indices(66, 1)]
# np.corrcoef(t3)[0, 1]
#
#
# import plotly.graph_objects as go  # for data visualisation
# import plotly.io as pio
#
# trace=go.Histogram(x=(w[np.triu_indices(66, 1)]/1e7), nbinsx=10000, histnorm="percent")
# fig=go.Figure(data=trace)
# pio.show(fig)