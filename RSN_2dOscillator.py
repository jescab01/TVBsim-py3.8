from tvb.simulator.lab import *
import numpy as np
from toolbox import timeseriesPlot, FFTplot, FFTpeak, AEC, PLV, PLI, epochingTool
import mne.filter
import scipy.signal

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = "FC_subj01"

simLength = 5500 # ms
samplingFreq = 1000 #Hz

m = models.Generic2dOscillator()
conn = connectivity.Connectivity.from_file("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\CTB_connx66_subj04.zip")
conn.weights=conn.weights/1e7
conn.speed=np.array([12])

coup = coupling.Linear(a=np.array([14.0]))
int = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-4]))) # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=int, monitors=mon)
sim.configure()

output = sim.run(simulation_length=simLength)
# Extract data
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels

# Plot raw time series
timeseriesPlot(raw_data, raw_time, regionLabels)
# FFourier peak
fft_peak = FFTpeak(raw_data,simLength)
# # Fourier Analysis plot
FFTplot(raw_data, simLength, regionLabels)


bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
for b in range(len(bands[0])):

    (lowcut, highcut) = bands[1][b]

    # Band-pass filtering
    filterSignals = mne.filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

    # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
    efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

    # Obtain Analytical signal
    efPhase = list()
    efEnvelope = list()
    for i in range(len(efSignals)):
        analyticalSignal = scipy.signal.hilbert(efSignals[i])
        # Get instantaneous phase and amplitude envelope by channel
        efPhase.append(np.unwrap(np.angle(analyticalSignal)))
        efEnvelope.append(np.abs(analyticalSignal))

    #CONNECTIVITY MEASURES
    ## PLV
    plv = PLV(efPhase)
    fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
    np.savetxt(fname, plv)

    ## PLI
    pli = PLI(efPhase)
    fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"pli"
    np.savetxt(fname, pli)
    ## AEC
    aec = AEC(efEnvelope)
    fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
    np.savetxt(fname, aec)

# # Copy structural connetivity weights in FC folder
weights=conn.weights
fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\weights.txt"
np.savetxt(fname, weights)

# # Load empirical data to make simple comparisons
# plv_emp = np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\3-alfaplv.txt")
# pli_emp = np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\3-alfapli.txt")
# aec_emp = np.loadtxt("C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\FC_subj04\\3-alfacorramp.txt")
#
# # Comparisons
# t1 = np.zeros(shape=(2, 2145))
# t1[0, :] = plv[np.triu_indices(66, 1)]
# t1[1, :] = plv_emp[np.triu_indices(66, 1)]
# plv_r = np.corrcoef(t1)[0, 1]
# # newRow.append(plv_r)
#
# t2 = np.zeros(shape=(2, 2145))
# t2[0, :] = pli[np.triu_indices(66, 1)]
# t2[1, :] = pli_emp[np.triu_indices(66, 1)]
# pli_r = np.corrcoef(t2)[0, 1]
# # newRow.append(pli_r)
#
# t3 = np.zeros(shape=(2, 2145))
# t3[0, :] = aec[np.triu_indices(66, 1)]
# t3[1, :] = aec_emp[np.triu_indices(66, 1)]
# aec_r = np.corrcoef(t3)[0, 1]
# # newRow.append(aec_r)

