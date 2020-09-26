import numpy as np
import time
from tvb.simulator.lab import *
from toolbox import epochingTool,bandpassFIRfilter,PLV, PLI, AEC, timeseriesPlot, FFTplot, plotConversions  # Home-made code
import mne.filter
import scipy.signal

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = "FC_subj1"

# Prepare simulation components
simLength = 1*1*1000 # ms
samplingFreq=1000 #Hz

m = models.ReducedWongWang(a=np.array(0.27), w=np.array(0.9), I_o=np.array(0.3), sigma_noise=np.array(0.001))
conn = connectivity.Connectivity.from_file("/CTB_data/output/CTB_connx66_subj04.zip")
conn.weights = conn.scaled_weights(mode="tract") # Scales by matrix's max value
coup = coupling.Linear(a=np.array(0.267))
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
int = integrators.EulerStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([1e-5])))
mon = (monitors.Raw(), monitors.TemporalAverage(period=1.0), )

# Run simulation
tic = time.time()
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=int, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)
print("simulation required %0.3f seconds.\n" % (time.time() - tic,))

# Extract data: "output[a][b][:,0,:,0].T" where:
# raw_data = output[0][1] = array w/ shape(t*p*l*o) (timepoints, variables of NMM, nodes, modes of NMM)
# a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
raw_data = output[0][1][1000:, 0, :, 0].T
raw_time = output[0][0][1000:]
regionLabels = conn.region_labels

del coup, int, m, models, mon, sim, tic

# # Plot raw time series
# timeseriesPlot(raw_data, raw_time, regionLabels)
#
# # Fourier Analysis plot
# FFTplot(raw_data, simLength, regionLabels)

bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
for b in range(len(bands[0])):
    (lowcut, highcut) = bands[1][b]

    # Band-pass filtering
    filterSignals = mne.filter.filter_data(raw_data,samplingFreq,lowcut,highcut)

    # Obtain Analytical signal
    analyticalSignal = scipy.signal.hilbert(filterSignals)
    # Get instantaneous phase and amplitude envelope by channel
    phase = np.unwrap(np.angle(analyticalSignal))
    amplitude_envelope = abs(analyticalSignal)

    # Plot conversions; as raw data is not centred around 0:
    # rdCentred = raw_data-np.asarray([np.average(raw_data,1),]*len(raw_data[0])).T
    # plotConversions(rdCentred, filterSignals, phase, amplitude_envelope, regionLabels, 1, raw_time)

    # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
    efPhase = epochingTool(phase, 4, samplingFreq, "phase")
    efEnvelope = epochingTool(amplitude_envelope, 4, samplingFreq, "envelope")

    # CONNECTIVITY MEASURES
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

# Copy structural connetivity weights in FC folder
weights=conn.weights
fname = "C:\\Users\\F_r_e\\PycharmProjects\\TVBsim-py3.8\\CTB_data\\output\\"+subjectid+"\\weights.txt"
np.savetxt(fname, weights)


# # aec1=mne.connectivity.envelope_correlation(filterSignals)
# x=np.reshape(plv[1])
# y=np.reshape(aev[1])     # No correlacionan en absoluto. ¿Es posible?
# np.corrcoef(plv[0, 1:],aec[0, 1:])
