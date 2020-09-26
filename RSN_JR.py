import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool, PLV, PLI, AEC, timeseriesPlot, FFTplot, plotConversions  # Home-made code
import mne.filter
import scipy.signal

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
# subjectid = "FC_subj1"

# Prepare simulation parameters
simLength = 4100 # ms
samplingFreq = 1000 #Hz

conn = connectivity.Connectivity.from_file("/CTB_data/output/CTB_connx66_subj04.zip")
# conn.weights = conn.weights/1e7
conn.weights=conn.scaled_weights(mode="region")
# data1 = conn.weights[np.triu_indices(66, 1)]
#
# conn = connectivity.Connectivity.from_file("connectivity_66.zip")
# data2 = conn.weights[np.triu_indices(66, 1)]
#
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# plt.hist(data1, bins=100, alpha=0.9, histtype='stepfilled', color='lightsteelblue',edgecolor='none', log=True);
# plt.hist(data2, bins=100, alpha=0.5, histtype='stepfilled', color='lightcoral',edgecolor='none', log=True);

#load FC (PLV) band alpha
plv_emp=np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\3-alfaplv.txt")

w = conn.weights[np.triu_indices(66, 1)]
fc = plv_emp[np.triu_indices(66, 1)]

import plotly.express as px
fig=px.scatter(x=w, y=fc)
fig.show()



mon = (monitors.Raw(), monitors.TemporalAverage(period=1.0))

jrm = models.JansenRit(A=np.array(5),B=np.array(19),mu=np.array(0.09), p_max=np.array(0.15), p_min=np.array(0.03), v0=np.array(6.0), r=np.array(0.56))
jrm.variables_of_interest=jrm.state_variables
phi_n_scaling = (jrm.a * jrm.A * (jrm.p_max-jrm.p_min) * 0.5) ** 2 / 2.
sigma = np.zeros(6)
sigma[3] = phi_n_scaling

int = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=sigma)) # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)

coup = coupling.SigmoidalJansenRit(a=np.array([0.1]))
# Run simulation
tic = time.time()
sim = simulator.Simulator(model=jrm, connectivity=conn, coupling=coup,  integrator=int, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds.\n" % (time.time() - tic,))

# Extract data
raw_data = output[0][1][:, :, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels

# Plot raw time series
timeseriesPlot(raw_data[:,1,:]-raw_data[:,2,:], raw_time, regionLabels)

# # # Fourier Analysis plot
# FFTplot(raw_data, simLength, regionLabels)
#
# bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
# #for b in range(len(bands[0])):
#     b=2
#     (lowcut, highcut) = bands[1][b]
#
#     # Band-pass filtering
#     filterSignals = mne.filter.filter_data(raw_data, samplingFreq, lowcut, highcut)
#
#     # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
#     efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")
#
#     # Obtain Analytical signal
#     efPhase = list()
#     efEnvelope = list()
#     for i in range(len(efSignals)):
#         analyticalSignal = scipy.signal.hilbert(efSignals[i])
#         # Get instantaneous phase and amplitude envelope by channel
#         efPhase.append(np.unwrap(np.angle(analyticalSignal)))
#         efEnvelope.append(np.abs(analyticalSignal))
#
#     # Check point
#     plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], regionLabels, 1, raw_time)

      # CONNECTIVITY MEASURES
#     ## PLV
#     plv = PLV(efPhase)
#     fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
#     np.savetxt(fname, plv)
#
    ## PLI
    # pli = PLI(efPhase)
    # fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"pli"
    # np.savetxt(fname, pli)
#
#     ## AEC
#     aec = AEC(efEnvelope)
#     fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
#     np.savetxt(fname, aec)
#
# # Copy structural connetivity weights in FC folder
# weights=conn.weights
# fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\weights.txt"
# np.savetxt(fname, weights)
#
#
# # # aec1=mne.connectivity.envelope_correlation(filterSignals)
# # x=np.reshape(plv[1])
# # y=np.reshape(aev[1])     # No correlacionan en absoluto. ¿Es posible?
# # np.corrcoef(plv[0, 1:],aec[0, 1:])
